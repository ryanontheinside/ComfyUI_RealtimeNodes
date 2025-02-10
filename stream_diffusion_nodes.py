import torch
from .base.control_base import ControlNodeBase
import comfy.model_management
import comfy.samplers
import random

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
            
            # Handle both batched and single sigmas
            sigma = args["sigma"]
            if torch.is_tensor(sigma):
                # For batched sampling, use first sigma in batch
                # This is safe because we process in order and track seen sigmas
                sigma = sigma[0].item() if len(sigma.shape) > 0 else sigma.item()
            
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


class StreamBatchSampler(ControlNodeBase):
    """Implements batched denoising for faster inference by processing multiple steps in parallel"""
    
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "update"
    CATEGORY = "real-time/sampling"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "batch_size": ("INT", {
                "default": 2,
                "min": 1,
                "max": 10,
                "step": 1,
                "tooltip": "Number of steps to batch together. Higher values use more memory but are faster."
            }),
        })
        return inputs
    
    def __init__(self):
        super().__init__()
        self.batch_size = None
    
    def sample(self, model, noise, sigmas, extra_args=None, callback=None, disable=None):
        """Sample with batched denoising steps"""
        extra_args = {} if extra_args is None else extra_args
        print(f"[StreamBatchSampler] Starting sampling with {len(sigmas)-1} steps, batch_size={self.batch_size}")
        print(f"[StreamBatchSampler] Input noise shape: {noise.shape}, device: {noise.device}")
        print(f"[StreamBatchSampler] Sigmas: {sigmas.tolist()}")
        
        # Prepare batched sampling
        num_sigmas = len(sigmas) - 1
        num_batches = (num_sigmas + self.batch_size - 1) // self.batch_size
        x = noise
        
        for batch_idx in range(num_batches):
            # Get sigmas for this batch
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_sigmas)
            batch_sigmas = sigmas[start_idx:end_idx+1]
            print(f"\n[StreamBatchSampler] Batch {batch_idx+1}/{num_batches}")
            print(f"[StreamBatchSampler] Processing steps {start_idx}-{end_idx}")
            print(f"[StreamBatchSampler] Batch sigmas: {batch_sigmas.tolist()}")
            
            # Create batch of identical latents
            batch_size = end_idx - start_idx
            x_batch = x.repeat(batch_size, 1, 1, 1)
            
            # Create batch of sigmas
            sigma_batch = batch_sigmas[:-1]  # All but last sigma
            
            # Run model on entire batch at once
            with torch.no_grad():
                # Process all steps in parallel
                denoised_batch = model(x_batch, sigma_batch, **extra_args)
                print(f"[StreamBatchSampler] Denoised batch shape: {denoised_batch.shape}")
                
                # Process results one at a time to maintain callback
                for i in range(batch_size):
                    sigma = sigma_batch[i]
                    sigma_next = batch_sigmas[i + 1]
                    denoised = denoised_batch[i:i+1]
                    
                    # Calculate step size (now always positive as we go from high to low sigma)
                    dt = sigma - sigma_next
                    
                    # Update x using Euler method
                    # The (denoised - x) term gives us the direction to move
                    # dt/sigma scales how far we move based on current noise level
                    x = x + (denoised - x) * (dt / sigma)
                    print(f"[StreamBatchSampler] Step {start_idx+i}: sigma={sigma:.4f}, next_sigma={sigma_next:.4f}, dt={dt:.4f}")
                    
                    # Call callback if provided
                    if callback is not None:
                        callback({'x': x, 'i': start_idx + i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})
        
        print(f"\n[StreamBatchSampler] Sampling complete. Final x shape: {x.shape}")
        return x
    
    def update(self, batch_size=2, always_execute=True):
        """Create sampler with specified settings"""
        self.batch_size = batch_size
        sampler = comfy.samplers.KSAMPLER(self.sample)
        return (sampler,)


class StreamScheduler(ControlNodeBase):
    """Implements StreamDiffusion's efficient timestep selection"""
    
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "update"
    CATEGORY = "real-time/sampling"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "model": ("MODEL",),
            "t_index_list": ("STRING", {
                "default": "32,45",
                "tooltip": "Comma-separated list of timesteps to actually use for denoising. Examples: '32,45' for img2img or '0,16,32,45' for txt2img"
            }),
            "num_inference_steps": ("INT", {
                "default": 50,
                "min": 1,
                "max": 1000,
                "step": 1,
                "tooltip": "Total number of timesteps in schedule. StreamDiffusion uses 50 by default. Only timesteps specified in t_index_list are actually used."
            }),
        })
        return inputs

    def update(self, model, t_index_list="32,45", num_inference_steps=50, always_execute=True):
        # Get model's sampling parameters
        model_sampling = model.get_model_object("model_sampling")
        print(f"[StreamScheduler] Model sampling max sigma: {model_sampling.sigma_max}, min sigma: {model_sampling.sigma_min}")
        
        # Parse timestep list
        try:
            t_index_list = [int(t.strip()) for t in t_index_list.split(",")]
        except ValueError as e:
            print(f"Error parsing timesteps: {e}. Using default [32,45]")
            t_index_list = [32, 45]
        print(f"[StreamScheduler] Using timesteps: {t_index_list}")
        
        # Create full schedule using normal scheduler
        full_sigmas = comfy.samplers.normal_scheduler(model_sampling, num_inference_steps)
        print(f"[StreamScheduler] Full sigma schedule: {full_sigmas.tolist()}")
        
        # Select only the sigmas at our desired indices, but in reverse order
        # This ensures we go from high noise to low noise
        selected_sigmas = []
        for t in sorted(t_index_list, reverse=True):  # Sort in reverse to go from high noise to low
            if t < 0 or t >= num_inference_steps:
                print(f"Warning: timestep {t} out of range [0,{num_inference_steps}), skipping")
                continue
            selected_sigmas.append(float(full_sigmas[t]))
        print(f"[StreamScheduler] Selected sigmas: {selected_sigmas}")
        
        # Add final sigma
        selected_sigmas.append(0.0)
        print(f"[StreamScheduler] Final sigma schedule: {selected_sigmas}")
        
        # Convert to tensor and move to appropriate device
        selected_sigmas = torch.FloatTensor(selected_sigmas).to(comfy.model_management.get_torch_device())
        return (selected_sigmas,)


class StreamCrossAttention(ControlNodeBase):
    """Implements optimized cross attention with KV-cache for real-time generation
    
    Paper reference: StreamDiffusion Section 3.5 "Pre-computation"
    - Pre-computes and caches prompt embeddings
    - Stores Key-Value pairs for reuse with static prompts
    - Only recomputes KV pairs when prompt changes
    
    Additional optimizations beyond paper:
    - QK normalization for better numerical stability
    - Rotary position embeddings (RoPE) for improved temporal consistency
    - Configurable context window size for memory/quality tradeoff
    """
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "update"
    CATEGORY = "real-time/sampling"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "model": ("MODEL",),
            "qk_norm": ("BOOLEAN", {
                "default": True,
                "tooltip": "Additional optimization: Whether to apply layer normalization to query and key tensors"
            }),
            "use_rope": ("BOOLEAN", {
                "default": True,
                "tooltip": "Additional optimization: Whether to use rotary position embeddings for better temporal consistency"
            }),
            "context_size": ("INT", {
                "default": 4,
                "min": 1,
                "max": 32,
                "step": 1,
                "tooltip": "Additional optimization: Maximum number of past frames to keep in context. Higher values use more memory but may improve temporal consistency."
            }),
            "use_kv_cache": ("BOOLEAN", {
                "default": True,
                "tooltip": "Paper Section 3.5: Whether to cache key-value pairs for static prompts to avoid recomputation"
            }),
        })
        return inputs
    
    def __init__(self):
        super().__init__()
        self.last_model_hash = None
        self.cross_attention_hook = None

    def update(self, model, always_execute=True, qk_norm=True, use_rope=True, context_size=4, use_kv_cache=True):
        print(f"[StreamCrossAttention] Initializing with qk_norm={qk_norm}, use_rope={use_rope}, context_size={context_size}, use_kv_cache={use_kv_cache}")
        
        # Get state with defaults
        state = self.get_state({
            "qk_norm": qk_norm,  # Additional optimization
            "use_rope": use_rope,  # Additional optimization
            "context_size": context_size,  # Additional optimization
            "use_kv_cache": use_kv_cache,  # From paper Section 3.5
            "workflow_count": 0,
            "context_queue": [],  # Additional: Store past context tensors for temporal consistency
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
                # Additional optimization: Temporal context management
                if len(state["context_queue"]) >= state["context_size"]:
                    state["context_queue"].pop(0)
                state["context_queue"].append(context.detach().clone())
                
                # Additional optimization: Use past context for temporal consistency
                full_context = torch.cat(state["context_queue"], dim=1)
                
                # Generate k/v for full context
                k = module.to_k(full_context)
                v = value if value is not None else module.to_v(full_context)
                
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
                full_seq_len = k.shape[1]  # Use full context length for k/v
                dim = q.shape[2]
                
                # Create position indices for q and k separately
                q_position = torch.arange(seq_len, device=q.device).unsqueeze(0).unsqueeze(-1)
                k_position = torch.arange(full_seq_len, device=k.device).unsqueeze(0).unsqueeze(-1)
                
                q_position = q_position.repeat(batch_size, 1, dim//2)
                k_position = k_position.repeat(batch_size, 1, dim//2)
                
                # Calculate frequencies
                freq = 10000.0 ** (-torch.arange(0, dim//2, 2, device=q.device) / dim)
                freq = freq.repeat((dim + 1) // 2)[:dim//2]
                
                # Calculate rotation angles
                q_theta = q_position * freq
                k_theta = k_position * freq
                
                # Apply rotations to q
                q_cos = torch.cos(q_theta)
                q_sin = torch.sin(q_theta)
                q_reshaped = q.view(*q.shape[:-1], -1, 2)
                q_out = torch.cat([
                    q_reshaped[..., 0] * q_cos - q_reshaped[..., 1] * q_sin,
                    q_reshaped[..., 0] * q_sin + q_reshaped[..., 1] * q_cos
                ], dim=-1)
                
                # Apply rotations to k
                k_cos = torch.cos(k_theta)
                k_sin = torch.sin(k_theta)
                k_reshaped = k.view(*k.shape[:-1], -1, 2)
                k_out = torch.cat([
                    k_reshaped[..., 0] * k_cos - k_reshaped[..., 1] * k_sin,
                    k_reshaped[..., 0] * k_sin + k_reshaped[..., 1] * k_cos
                ], dim=-1)
                
                q = q_out
                k = k_out
            
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




