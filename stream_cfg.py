import torch
from .base.control_base import ControlNodeBase
import comfy.model_management
import comfy.samplers
import random
import math

class StreamCFG(ControlNodeBase):
    """Implements CFG approaches for temporal consistency between workflow runs"""
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "update"
    CATEGORY = "real-time/sampling"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        del inputs["required"]["always_execute"]
        inputs["required"].update({
            "model": ("MODEL",),
            "cfg_type": (["self", "full", "initialize"], {
                "default": "self",
                "tooltip": "Type of CFG to use: full (standard), self (memory efficient), or initialize (memory efficient with initialization)"
            }),
            "residual_scale": ("FLOAT", {
                "default": 0.7,
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
        self.last_model_hash = None
        self.post_cfg_function = None

    def update(self, model, always_execute=True, cfg_type="self", residual_scale=0.7, delta=1.0):
        print(f"[StreamCFG] Initializing with cfg_type={cfg_type}, residual_scale={residual_scale}, delta={delta}")
        
        state = self.get_state({
            "last_uncond": None,
            "initialized": False,
            "cfg_type": cfg_type,
            "residual_scale": residual_scale,
            "delta": delta,
            "workflow_count": 0,
            "current_sigmas": None,
            "seen_sigmas": set(),
            "is_last_step": False,
            "alpha_prod_t": None,
            "beta_prod_t": None,
            "c_skip": None,
            "c_out": None,
            "last_noise": None,  # Store noise from previous frame
        })
        
        def post_cfg_function(args):
            denoised = args["denoised"]
            cond = args["cond"]
            uncond = args["uncond"]
            cond_denoised = args["cond_denoised"]
            uncond_denoised = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            
            print(f"\n[StreamCFG Debug] Step Info:")
            print(f"- Workflow count: {state['workflow_count']}")
            print(f"- CFG Type: {state['cfg_type']}")
            print(f"- Tensor Stats:")
            print(f"  - denoised shape: {denoised.shape}, range: [{denoised.min():.3f}, {denoised.max():.3f}]")
            print(f"  - uncond_denoised shape: {uncond_denoised.shape}, range: [{uncond_denoised.min():.3f}, {uncond_denoised.max():.3f}]")
            if state["last_uncond"] is not None:
                print(f"  - last_uncond shape: {state['last_uncond'].shape}, range: [{state['last_uncond'].min():.3f}, {state['last_uncond'].max():.3f}]")
            
            sigma = args["sigma"]
            if torch.is_tensor(sigma):
                sigma = sigma[0].item() if len(sigma.shape) > 0 else sigma.item()
            print(f"- Current sigma: {sigma:.6f}")
            
            model_options = args["model_options"]
            sample_sigmas = model_options["transformer_options"].get("sample_sigmas", None)
            
            if sample_sigmas is not None and state["current_sigmas"] is None:
                sigmas = [s.item() for s in sample_sigmas]
                if sigmas[-1] == 0.0:
                    sigmas = sigmas[:-1]
                state["current_sigmas"] = sigmas
                state["seen_sigmas"] = set()
                print(f"- New sigma sequence: {sigmas}")
                
                state["alpha_prod_t"] = torch.tensor([1.0 / (1.0 + s**2) for s in sigmas], 
                    device=denoised.device, dtype=denoised.dtype)
                state["beta_prod_t"] = torch.tensor([s / (1.0 + s**2) for s in sigmas],
                    device=denoised.device, dtype=denoised.dtype)
                
                state["c_skip"] = torch.tensor([1.0 / (s**2 + 1.0) for s in sigmas],
                    device=denoised.device, dtype=denoised.dtype)
                state["c_out"] = torch.tensor([-s / torch.sqrt(torch.tensor(s**2 + 1.0)) for s in sigmas],
                    device=denoised.device, dtype=denoised.dtype)
                
                print(f"- Scaling factors for first step:")
                print(f"  alpha: {state['alpha_prod_t'][0]:.6f}")
                print(f"  beta: {state['beta_prod_t'][0]:.6f}")
                print(f"  c_skip: {state['c_skip'][0]:.6f}")
                print(f"  c_out: {state['c_out'][0]:.6f}")
                
                # Initialize noise for first step using previous frame if available
                if state["last_uncond"] is not None and state["last_noise"] is not None:
                    # Scale noise based on current sigma
                    current_sigma = torch.tensor(sigmas[0], device=denoised.device, dtype=denoised.dtype)
                    scaled_noise = state["last_noise"] * current_sigma
                    
                    # Mix with previous frame prediction
                    alpha = 1.0 / (1.0 + current_sigma**2)
                    noisy_input = alpha * state["last_uncond"] + (1 - alpha) * scaled_noise
                    
                    # Update model input
                    if "input" in model_options:
                        model_options["input"] = noisy_input
                    print(f"- Initialized with previous frame, noise scale: {current_sigma:.6f}")
            
            state["seen_sigmas"].add(sigma)
            state["is_last_step"] = False
            if state["current_sigmas"] is not None:
                is_last_step = len(state["seen_sigmas"]) >= len(state["current_sigmas"])
                if not is_last_step and sigma == min(state["current_sigmas"]):
                    is_last_step = True
                state["is_last_step"] = is_last_step
                print(f"- Is last step: {is_last_step}")
                print(f"- Seen sigmas: {sorted(state['seen_sigmas'])}")
            
            # First workflow case
            if state["last_uncond"] is None:
                if state["is_last_step"]:
                    state["last_uncond"] = uncond_denoised.detach().clone()
                    # Store noise for next frame initialization
                    if "noise" in args:
                        state["last_noise"] = args["noise"].detach().clone()
                    state["workflow_count"] += 1
                    state["current_sigmas"] = None
                    if cfg_type == "initialize":
                        state["initialized"] = True
                    self.set_state(state)
                    print("- First workflow complete, stored last_uncond and noise")
                return denoised
            
            current_idx = len(state["seen_sigmas"]) - 1
            print(f"- Current step index: {current_idx}")
            
            # Apply temporal consistency at first step and blend throughout
            if current_idx == 0:
                # Strong influence at first step
                noise_pred_uncond = state["last_uncond"] * state["delta"]
                result = noise_pred_uncond + cond_scale * (cond_denoised - noise_pred_uncond)
                # Apply residual scale to entire result for stronger consistency
                result = result * state["residual_scale"] + denoised * (1 - state["residual_scale"])
            else:
                # Lighter influence in later steps
                blend_scale = state["residual_scale"] * (1 - current_idx / len(state["current_sigmas"]))
                result = denoised * (1 - blend_scale) + uncond_denoised * blend_scale
            
            print(f"- Result range after blending: [{result.min():.3f}, {result.max():.3f}]")
            
            # Store last prediction if this is the last step
            if state["is_last_step"]:
                state["last_uncond"] = uncond_denoised.detach().clone()
                # Store noise for next frame initialization
                if "noise" in args:
                    state["last_noise"] = args["noise"].detach().clone()
                state["workflow_count"] += 1
                state["current_sigmas"] = None
                self.set_state(state)
                print(f"- Stored new last_uncond range: [{state['last_uncond'].min():.3f}, {state['last_uncond'].max():.3f}]")
            
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


