import torch
from .base.control_base import ControlNodeBase

class StreamCFG(ControlNodeBase):
    """Implements CFG approaches for temporal consistency between workflow runs"""
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "update"
    CATEGORY = "real-time/sampling"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "model": ("MODEL",),
            "cfg_type": (["self", "full", "initialize"], {
                "default": "self",
                "tooltip": "Type of CFG to use: full (standard), self (memory efficient), or initialize (memory efficient with initialization)"
            }),
            "residual_scale": ("FLOAT", {
                "default": 0.4,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Scale factor for residual (higher = more temporal consistency)"
            }),
            "delta": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 5.0,
                "step": 0.1,
                "tooltip": "Delta parameter for self/initialize CFG types"
            }),
        })
        return inputs
    
    def __init__(self):
        super().__init__()
        # Store the last model to detect when we need to reapply the hook
        self.last_model_hash = None
        self.post_cfg_function = None

    def update(self, model, always_execute=True, cfg_type="self", residual_scale=0.4, delta=1.0):
        print(f"[StreamCFG] Initializing with cfg_type={cfg_type}, residual_scale={residual_scale}, delta={delta}")
        
        # Get state with defaults
        state = self.get_state({
            "last_uncond": None,  # Store last workflow's unconditioned prediction
            "initialized": False,
            "cfg_type": cfg_type,
            "residual_scale": residual_scale,
            "delta": delta,
            "workflow_count": 0,  # Track number of workflow runs
            "current_sigmas": None,  # Track sigmas for this workflow
            "seen_sigmas": set(),  # Track which sigmas we've seen this workflow
            "is_last_step": False,  # Track if we're on the last step
            "alpha_prod_t": None,  # Store alpha values for proper scaling
            "beta_prod_t": None,  # Store beta values for proper scaling
        })
        
        def post_cfg_function(args):
            # Extract info
            denoised = args["denoised"]
            cond = args["cond"]
            uncond = args["uncond"]
            cond_denoised = args["cond_denoised"]
            uncond_denoised = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            sigma = args["sigma"].item() if torch.is_tensor(args["sigma"]) else args["sigma"]
            
            # Get step info from model options
            model_options = args["model_options"]
            sample_sigmas = model_options["transformer_options"].get("sample_sigmas", None)
            
            # Update current sigmas if needed
            if sample_sigmas is not None and state["current_sigmas"] is None:
                # Filter out the trailing 0.0 if present
                sigmas = [s.item() for s in sample_sigmas]
                if sigmas[-1] == 0.0:
                    sigmas = sigmas[:-1]
                state["current_sigmas"] = sigmas
                state["seen_sigmas"] = set()
                
                # Calculate alpha and beta values for proper scaling
                alphas = [1.0 / (1.0 + s**2) for s in sigmas]
                state["alpha_prod_t"] = torch.tensor(alphas, device=denoised.device, dtype=denoised.dtype)
                state["beta_prod_t"] = torch.sqrt(1 - state["alpha_prod_t"])
            
            # Track this sigma
            state["seen_sigmas"].add(sigma)
            
            # Check if this is the last step
            state["is_last_step"] = False
            if state["current_sigmas"] is not None:
                # It's the last step if we've seen all sigmas
                is_last_step = len(state["seen_sigmas"]) >= len(state["current_sigmas"])
                # Or if this is the smallest sigma in the sequence
                if not is_last_step and sigma == min(state["current_sigmas"]):
                    is_last_step = True
                state["is_last_step"] = is_last_step
            
            # First workflow case
            if state["last_uncond"] is None:
                if state["is_last_step"]:
                    state["last_uncond"] = uncond_denoised.detach().clone()
                    state["workflow_count"] += 1
                    state["current_sigmas"] = None  # Reset for next workflow
                    if cfg_type == "initialize":
                        state["initialized"] = True
                    self.set_state(state)
                return denoised
            
            # Handle different CFG types for subsequent workflows
            if cfg_type == "full":
                result = denoised
                
            elif cfg_type == "initialize" and not state["initialized"]:
                result = denoised
                if state["is_last_step"]:
                    state["initialized"] = True
                    self.set_state(state)
                
            else:  # self or initialized initialize
                # Get current step index
                current_idx = len(state["seen_sigmas"]) - 1
                
                # Scale last prediction with proper alpha/beta values
                noise_pred_uncond = state["last_uncond"] * delta
                
                # Apply CFG with scaled prediction
                result = noise_pred_uncond + cond_scale * (cond_denoised - noise_pred_uncond)
                
                # Store last prediction if this is the last step
                if state["is_last_step"]:
                    # Calculate properly scaled residual
                    scaled_noise = state["beta_prod_t"][current_idx] * state["last_uncond"]
                    delta_x = uncond_denoised - scaled_noise
                    
                    # Scale delta_x with next step's alpha/beta
                    if current_idx < len(state["current_sigmas"]) - 1:
                        alpha_next = state["alpha_prod_t"][current_idx + 1]
                        beta_next = state["beta_prod_t"][current_idx + 1]
                    else:
                        alpha_next = torch.ones_like(state["alpha_prod_t"][0])
                        beta_next = torch.ones_like(state["beta_prod_t"][0])
                    
                    delta_x = alpha_next * delta_x / beta_next
                    
                    # Update stored prediction with scaled residual
                    final_update = uncond_denoised + residual_scale * delta_x
                    state["last_uncond"] = final_update
                    state["workflow_count"] += 1
                    state["current_sigmas"] = None  # Reset for next workflow
                    self.set_state(state)
            
            return result

        # Store function reference to prevent garbage collection
        self.post_cfg_function = post_cfg_function

        # Only set up post CFG function if model has changed
        model_hash = hash(str(model))
        if model_hash != self.last_model_hash:
            m = model.clone()
            m.model_options = m.model_options.copy()
            m.model_options["sampler_post_cfg_function"] = [self.post_cfg_function]
            self.last_model_hash = model_hash
            return (m,)
        
        # Make sure our function is still in the list
        if not any(f is self.post_cfg_function for f in model.model_options.get("sampler_post_cfg_function", [])):
            m = model.clone()
            m.model_options = m.model_options.copy()
            m.model_options["sampler_post_cfg_function"] = [self.post_cfg_function]
            return (m,)
        
        return (model,)


class StreamConditioning(ControlNodeBase):
    """Applies Residual CFG to conditioning for improved temporal consistency with different CFG types"""
    #NOTE: experimental
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg_type": (["full", "self", "initialize"], {
                    "default": "full",
                    "tooltip": "Type of CFG to use: full (standard), self (memory efficient), or initialize (memory efficient with initialization)"
                }),
                "residual_scale": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Scale factor for residual conditioning (higher = more temporal consistency)"
                }),
                "delta": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Delta parameter for self/initialize CFG types"
                }),
                "always_execute": ("BOOLEAN", {
                    "default": False,
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "update"
    CATEGORY = "real-time/control/utility"

    def __init__(self):
        super().__init__()

    def update(self, positive, negative, cfg_type="full", residual_scale=0.4, delta=1.0, always_execute=False):
        # Get state with defaults
        state = self.get_state({
            "prev_positive": None,
            "prev_negative": None,
            "stock_noise": None,  # For self/initialize CFG
            "initialized": False  # For initialize CFG
        })

        # Extract conditioning tensors
        current_pos_cond = positive[0][0]  # Assuming standard ComfyUI conditioning format
        current_neg_cond = negative[0][0]
        
        # First frame case
        if state["prev_positive"] is None:
            state["prev_positive"] = current_pos_cond.detach().clone()
            state["prev_negative"] = current_neg_cond.detach().clone()
            if cfg_type == "initialize":
                # For initialize, we use the first negative as our stock noise
                state["stock_noise"] = current_neg_cond.detach().clone()
            elif cfg_type == "self":
                # For self, we start with a scaled version of the negative
                state["stock_noise"] = current_neg_cond.detach().clone() * delta
            self.set_state(state)
            return (positive, negative)

        # Handle different CFG types
        if cfg_type == "full":
            # Standard R-CFG with full negative conditioning
            pos_residual = current_pos_cond - state["prev_positive"]
            neg_residual = current_neg_cond - state["prev_negative"]
            
            blended_pos = current_pos_cond + residual_scale * pos_residual
            blended_neg = current_neg_cond + residual_scale * neg_residual
            
            # Update state
            state["prev_positive"] = current_pos_cond.detach().clone()
            state["prev_negative"] = current_neg_cond.detach().clone()
            
            # Reconstruct conditioning format
            positive_out = [[blended_pos, positive[0][1]]]
            negative_out = [[blended_neg, negative[0][1]]]
            
        else:  # self or initialize
            # Calculate residual for positive conditioning
            pos_residual = current_pos_cond - state["prev_positive"]
            blended_pos = current_pos_cond + residual_scale * pos_residual
            
            # Update stock noise based on current prediction
            if cfg_type == "initialize" and not state["initialized"]:
                # First prediction for initialize type
                state["stock_noise"] = current_neg_cond.detach().clone()
                state["initialized"] = True
            else:
                # Update stock noise with temporal consistency
                stock_residual = current_neg_cond - state["stock_noise"]
                state["stock_noise"] = current_neg_cond + residual_scale * stock_residual
            
            # Scale stock noise by delta
            scaled_stock = state["stock_noise"] * delta
            
            # Update state
            state["prev_positive"] = current_pos_cond.detach().clone()
            state["prev_negative"] = scaled_stock.detach().clone()
            
            # Reconstruct conditioning format
            positive_out = [[blended_pos, positive[0][1]]]
            negative_out = [[scaled_stock, negative[0][1]]]

        self.set_state(state)
        return (positive_out, negative_out)



