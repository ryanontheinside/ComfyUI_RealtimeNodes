import torch
from .base.control_base import ControlNodeBase
import comfy.model_management
import comfy.samplers
import random

class StreamCFG(ControlNodeBase):
    
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
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "update"
    CATEGORY = "real-time/sampling"
    DESCRIPTION = "Implements CFG approaches as seen in StreamDiffusion for temporal consistency between workflow runs."
    
    def __init__(self):
        super().__init__()
        # Store the last model to detect when we need to reapply the hook
        self.last_model_hash = None
        self.post_cfg_function = None

    def update(self, model, always_execute=True, cfg_type="self", residual_scale=0.4, delta=1.0):
        print(f"[StreamCFG] Initializing with cfg_type={cfg_type}, residual_scale={residual_scale}, delta={delta}")
        
        # Get state with defaults
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
            # Add new state variables for proper scaling
            "alpha_prod_t": None,
            "beta_prod_t": None,
            "c_skip": None,
            "c_out": None,
        })
        
        def post_cfg_function(args):
            # Extract info
            denoised = args["denoised"]
            cond = args["cond"]
            uncond = args["uncond"]
            cond_denoised = args["cond_denoised"]
            uncond_denoised = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            
            # Debug prints for tensor stats
            print(f"\n[StreamCFG Debug] Step Info:")
            print(f"- Workflow count: {state['workflow_count']}")
            print(f"- CFG Type: {state['cfg_type']}")
            print(f"- Tensor Stats:")
            print(f"  - denoised shape: {denoised.shape}, range: [{denoised.min():.3f}, {denoised.max():.3f}]")
            print(f"  - uncond_denoised shape: {uncond_denoised.shape}, range: [{uncond_denoised.min():.3f}, {uncond_denoised.max():.3f}]")
            if state["last_uncond"] is not None:
                print(f"  - last_uncond shape: {state['last_uncond'].shape}, range: [{state['last_uncond'].min():.3f}, {state['last_uncond'].max():.3f}]")
            
            # Handle both batched and single sigmas
            sigma = args["sigma"]
            if torch.is_tensor(sigma):
                sigma = sigma[0].item() if len(sigma.shape) > 0 else sigma.item()
            print(f"- Current sigma: {sigma:.6f}")
            
            # Get step info from model options
            model_options = args["model_options"]
            sample_sigmas = model_options["transformer_options"].get("sample_sigmas", None)
            
            # Update current sigmas if needed
            if sample_sigmas is not None and state["current_sigmas"] is None:
                sigmas = [s.item() for s in sample_sigmas]
                if sigmas[-1] == 0.0:
                    sigmas = sigmas[:-1]
                state["current_sigmas"] = sigmas
                state["seen_sigmas"] = set()
                print(f"- New sigma sequence: {sigmas}")
                
                # Calculate paper's exact scaling factors
                state["alpha_prod_t"] = torch.tensor([1.0 / (1.0 + s**2) for s in sigmas], 
                    device=denoised.device, dtype=denoised.dtype)
                state["beta_prod_t"] = torch.tensor([s / (1.0 + s**2) for s in sigmas],
                    device=denoised.device, dtype=denoised.dtype)
                
                # Calculate c_skip and c_out coefficients
                state["c_skip"] = torch.tensor([1.0 / (s**2 + 1.0) for s in sigmas],
                    device=denoised.device, dtype=denoised.dtype)
                state["c_out"] = torch.tensor([-s / torch.sqrt(torch.tensor(s**2 + 1.0)) for s in sigmas],
                    device=denoised.device, dtype=denoised.dtype)
                
                print(f"- Scaling factors for first step:")
                print(f"  alpha: {state['alpha_prod_t'][0]:.6f}")
                print(f"  beta: {state['beta_prod_t'][0]:.6f}")
                print(f"  c_skip: {state['c_skip'][0]:.6f}")
                print(f"  c_out: {state['c_out'][0]:.6f}")
            
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
                print(f"- Is last step: {is_last_step}")
                print(f"- Seen sigmas: {sorted(state['seen_sigmas'])}")
            
            # First workflow case
            if state["last_uncond"] is None:
                if state["is_last_step"]:
                    state["last_uncond"] = uncond_denoised.detach().clone()
                    state["workflow_count"] += 1
                    state["current_sigmas"] = None
                    if cfg_type == "initialize":
                        state["initialized"] = True
                    self.set_state(state)
                    print("- First workflow complete, stored last_uncond")
                return denoised
            
            # Handle different CFG types
            if cfg_type == "full":
                result = denoised
            elif cfg_type == "initialize" and not state["initialized"]:
                result = denoised
                if state["is_last_step"]:
                    state["initialized"] = True
                    self.set_state(state)
            else:  # self or initialized initialize
                current_idx = len(state["seen_sigmas"]) - 1
                print(f"- Current step index: {current_idx}")
                
                # Use paper's exact formulation for noise prediction
                noise_pred_uncond = state["last_uncond"] * state["delta"]
                print(f"- Scaled noise prediction range: [{noise_pred_uncond.min():.3f}, {noise_pred_uncond.max():.3f}]")
                
                # Apply CFG with scaled prediction
                result = noise_pred_uncond + cond_scale * (cond_denoised - noise_pred_uncond) * state["residual_scale"]
                print(f"- Result range after CFG: [{result.min():.3f}, {result.max():.3f}]")
                
                # Store last prediction if this is the last step
                if state["is_last_step"]:
                    # Calculate F_theta using paper's formulation
                    F_theta = (uncond_denoised - state["beta_prod_t"][current_idx] * noise_pred_uncond) / state["alpha_prod_t"][current_idx]
                    print(f"- F_theta range: [{F_theta.min():.3f}, {F_theta.max():.3f}]")
                    
                    delta_x = state["c_out"][current_idx] * F_theta + state["c_skip"][current_idx] * uncond_denoised
                    print(f"- delta_x range: [{delta_x.min():.3f}, {delta_x.max():.3f}]")
                    
                    # Scale delta_x with next step's coefficients
                    if current_idx < len(state["current_sigmas"]) - 1:
                        next_alpha = state["alpha_prod_t"][current_idx + 1]
                        next_beta = state["beta_prod_t"][current_idx + 1]
                    else:
                        next_alpha = torch.ones_like(state["alpha_prod_t"][0])
                        next_beta = torch.zeros_like(state["beta_prod_t"][0])
                    print(f"- Next step coefficients - alpha: {next_alpha:.6f}, beta: {next_beta:.6f}")
                    
                    # Update stored prediction with properly scaled residual
                    if next_beta > 0:
                        final_update = (next_alpha * delta_x) / next_beta
                        # Add noise only when beta > 0
                        noise = torch.randn_like(delta_x) * (1 - next_alpha**2).sqrt()
                        final_update = final_update + noise
                        print(f"- Added noise range: [{noise.min():.3f}, {noise.max():.3f}]")
                    else:
                        # For the last step, just use the current prediction
                        final_update = uncond_denoised
                    
                    print(f"- Final update range: [{final_update.min():.3f}, {final_update.max():.3f}]")
                    state["last_uncond"] = final_update
                    state["workflow_count"] += 1
                    state["current_sigmas"] = None
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


#NOTE: totally and utterly experimental. No theoretical backing whatsoever.
class StreamConditioning(ControlNodeBase):
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
                "context_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Number of past conditionings to keep in context. Higher values = smoother transitions but more memory usage."
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
    DESCRIPTION = "Applies Residual CFG to conditioning for improved temporal consistency with different CFG types. This is totally and utterly experimental. No theoretical backing whatsoever."

    def __init__(self):
        super().__init__()

    def update(self, positive, negative, cfg_type="full", residual_scale=0.4, delta=1.0, context_size=4, always_execute=False):
        # Get state with defaults
        state = self.get_state({
            "prev_positive": None,
            "prev_negative": None,
            "stock_noise": None,  # For self/initialize CFG
            "initialized": False,  # For initialize CFG
            "pos_context": [],    # Store past positive conditionings
            "neg_context": []     # Store past negative conditionings
        })

        # Extract conditioning tensors
        current_pos_cond = positive[0][0]
        current_neg_cond = negative[0][0]
        
        # Update context queues
        if len(state["pos_context"]) >= context_size:
            state["pos_context"].pop(0)
            state["neg_context"].pop(0)
        state["pos_context"].append(current_pos_cond.detach().clone())
        state["neg_context"].append(current_neg_cond.detach().clone())
        
        # First frame case
        if state["prev_positive"] is None:
            state["prev_positive"] = current_pos_cond.detach().clone()
            state["prev_negative"] = current_neg_cond.detach().clone()
            if cfg_type == "initialize":
                state["stock_noise"] = current_neg_cond.detach().clone()
            elif cfg_type == "self":
                state["stock_noise"] = current_neg_cond.detach().clone() * delta
            self.set_state(state)
            return (positive, negative)

        # Handle different CFG types
        if cfg_type == "full":
            # Use entire context for smoother transitions
            pos_context = torch.stack(state["pos_context"], dim=0)
            neg_context = torch.stack(state["neg_context"], dim=0)
            
            # Calculate weighted residuals across context
            weights = torch.linspace(0.5, 1.0, len(state["pos_context"]), device=current_pos_cond.device)
            pos_residual = (current_pos_cond - pos_context) * weights.view(-1, 1, 1)
            neg_residual = (current_neg_cond - neg_context) * weights.view(-1, 1, 1)
            
            # Average residuals
            pos_residual = pos_residual.mean(dim=0)
            neg_residual = neg_residual.mean(dim=0)
            
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

class StreamCrossAttention(ControlNodeBase):
    """Implements optimized cross attention with KV-cache for real-time generation
    
    Core functionality from StreamDiffusion:
    - Pre-computes and caches prompt embeddings
    - Stores Key-Value pairs for reuse with static prompts
    - Only recomputes KV pairs when prompt changes
    
    Additional optimizations beyond StreamDiffusion:
    - QK normalization for better numerical stability (disabled by default)
    - Rotary position embeddings (RoPE) for improved temporal consistency (disabled by default)
    """
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "update"
    CATEGORY = "real-time/sampling"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "model": ("MODEL",),
            # Core StreamDiffusion functionality
            "use_kv_cache": ("BOOLEAN", {
                "default": True,
                "tooltip": "StreamDiffusion: Whether to cache key-value pairs for static prompts to avoid recomputation"
            }),
            # Additional optimizations (all disabled by default)
            "qk_norm": ("BOOLEAN", {
                "default": False,
                "tooltip": "Additional optimization: Whether to apply layer normalization to query and key tensors"
            }),
            "use_rope": ("BOOLEAN", {
                "default": False,
                "tooltip": "Additional optimization: Whether to use rotary position embeddings for better temporal consistency"
            }),
        })
        return inputs
    
    def __init__(self):
        super().__init__()
        self.last_model_hash = None
        self.cross_attention_hook = None

    def update(self, model, always_execute=True, qk_norm=True, use_rope=True, use_kv_cache=True):
        print(f"[StreamCrossAttention] Initializing with qk_norm={qk_norm}, use_rope={use_rope}, use_kv_cache={use_kv_cache}")
        
        # Get state with defaults
        state = self.get_state({
            "qk_norm": qk_norm,  # Additional optimization
            "use_rope": use_rope,  # Additional optimization
            "use_kv_cache": use_kv_cache,  # From paper Section 3.5
            "workflow_count": 0,
            "kv_cache": {},  # From paper Section 3.5: Cache KV pairs for each prompt
            "last_prompt_embeds": None,  # From paper Section 3.5: For cache validation
        })
        
        def cross_attention_forward(module, x, context=None, mask=None, value=None):
            q = module.to_q(x)
            context = x if context is None else context
            
            # Paper Section 3.5: KV Caching Logic
            cache_hit = False
            if state["use_kv_cache"] and state["last_prompt_embeds"] is not None:
                # Compare current context with cached prompt embeddings
                if torch.allclose(context, state["last_prompt_embeds"], rtol=1e-5, atol=1e-5):
                    cache_hit = True
                    k, v = state["kv_cache"].get(module, (None, None))
                    if k is not None and v is not None:
                        print("[StreamCrossAttention] Using cached KV pairs")
            
            if not cache_hit:
                # Generate k/v for current context
                k = module.to_k(context)
                v = value if value is not None else module.to_v(context)
                
                # Paper Section 3.5: Cache KV pairs for static prompts
                if state["use_kv_cache"]:
                    state["last_prompt_embeds"] = context.detach().clone()
                    state["kv_cache"][module] = (k.detach().clone(), v.detach().clone())
            
            # Additional optimization: QK normalization
            if state["qk_norm"]:
                q_norm = torch.nn.LayerNorm(q.shape[-1], device=q.device, dtype=q.dtype)
                k_norm = torch.nn.LayerNorm(k.shape[-1], device=k.device, dtype=k.dtype)
                q = q_norm(q)
                k = k_norm(k)
            
            # Additional optimization: Rotary position embeddings
            if state["use_rope"]:
                # Calculate position embeddings
                batch_size = q.shape[0]
                seq_len = q.shape[1]
                dim = q.shape[2]
                
                # Create position indices
                position = torch.arange(seq_len, device=q.device).unsqueeze(0).unsqueeze(-1)
                position = position.repeat(batch_size, 1, dim//2)
                
                # Calculate frequencies
                freq = 10000.0 ** (-torch.arange(0, dim//2, 2, device=q.device) / dim)
                freq = freq.repeat((dim + 1) // 2)[:dim//2]
                
                # Calculate rotation angles
                theta = position * freq
                
                # Apply rotations to q and k
                cos = torch.cos(theta)
                sin = torch.sin(theta)
                
                def rotate_half(x):
                    x = x.view(*x.shape[:-1], -1, 2)
                    return torch.cat([
                        x[..., 0] * cos - x[..., 1] * sin,
                        x[..., 0] * sin + x[..., 1] * cos
                    ], dim=-1)
                
                q = rotate_half(q)
                k = rotate_half(k)
            
            # Standard attention computation with memory-efficient access pattern
            batch_size = q.shape[0]
            q_seq_len = q.shape[1]
            k_seq_len = k.shape[1]
            head_dim = q.shape[-1] // module.heads
            
            # Reshape for multi-head attention
            q = q.view(batch_size, q_seq_len, module.heads, head_dim)
            k = k.view(batch_size, k_seq_len, module.heads, head_dim)
            v = v.view(batch_size, k_seq_len, module.heads, head_dim)
            
            # Transpose for attention computation
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Compute attention scores
            scale = head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            if mask is not None:
                scores = scores + mask
            
            # Apply attention
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            
            # Reshape back
            out = out.transpose(1, 2).contiguous()
            out = out.view(batch_size, q_seq_len, -1)
            
            # Project back to original dimension
            out = module.to_out[0](out)
            
            return out
        
        def hook_cross_attention(module, input, output):
            if isinstance(module, torch.nn.Module) and hasattr(module, "to_q"):
                # Store original forward
                if not hasattr(module, "_original_forward"):
                    module._original_forward = module.forward
                # Replace with our optimized version
                module.forward = lambda *args, **kwargs: cross_attention_forward(module, *args, **kwargs)
            return output
        
        # Only set up hooks if model has changed
        model_hash = hash(str(model))
        if model_hash != self.last_model_hash:
            m = model.clone()
            
            # Remove old hooks if they exist
            if self.cross_attention_hook is not None:
                self.cross_attention_hook.remove()
            
            # Register hook for cross attention modules
            def register_hooks(module):
                if isinstance(module, torch.nn.Module) and hasattr(module, "to_q"):
                    self.cross_attention_hook = module.register_forward_hook(hook_cross_attention)
            
            m.model.apply(register_hooks)
            self.last_model_hash = model_hash
            return (m,)
        
        return (model,)




