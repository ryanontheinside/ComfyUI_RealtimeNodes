import torch
from .base.control_base import ControlNodeBase
import comfy.model_management
import comfy.samplers
import random
import math


class StreamBatchSampler(ControlNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "latent": ("LATENT",),
            "num_timesteps": ("INT", {
                "default": 2,
                "min": 1,
                "max": 10,
                "step": 1,
                "tooltip": "Number of denoising steps to use. More steps = better quality but MUCH slower. StreamDiffusion's speed comes from using just 2 steps for img2img (32,45) or 4 for txt2img (0,16,32,45)"
            }),
            "frame_buffer_size": ("INT", {
                "default": 1,
                "min": 1,
                "max": 10,
                "step": 1,
                "tooltip": "How many frames to process together for temporal consistency. Each frame is processed at each timestep (total batch = frame_buffer_size * num_timesteps). Higher values reduce flickering but use more VRAM and don't improve speed"
            }),
            "cfg_type": (["none", "full", "self", "initialize"], {
                "default": "self",
                "tooltip": "'self' is fastest and most memory efficient, 'full' is standard SD behavior but slower, 'initialize' is a middle ground, 'none' disables guidance"
            }),
            "guidance_scale": ("FLOAT", {
                "default": 1.2,
                "min": 1.0,
                "max": 20.0,
                "step": 0.1,
                "tooltip": "How closely to follow the prompt. StreamDiffusion works best with low values (1.2-1.5). Higher values = stronger prompt influence but more artifacts"
            }),
            "delta": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 5.0,
                "step": 0.1,
                "tooltip": "Only used with 'self' CFG. Controls strength of self-guidance. Higher values = stronger guidance but more artifacts. Default 1.0 works well"
            }),
            "similarity_threshold": ("FLOAT", {
                "default": 0.98,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Skip frames that are too similar to save compute. 0 = process all frames. Try 0.98 if you want to skip static scenes. Higher = more skipping"
                #TODO: see if this is backwards from original
            }),
            "max_skip_frames": ("INT", {
                "default": 10,
                "min": 1,
                "max": 100,
                "step": 1,
                "tooltip": "Maximum frames to skip when using similarity filter. Prevents getting stuck on static scenes. Lower = more responsive to changes"
            }),
        })
        inputs["optional"] = {
            "prompt": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Optional text prompt. Enables KV caching for faster processing. Leave empty to disable. Same prompt across frames = faster"
            }),
        }
        return inputs
    
    RETURN_TYPES = ("SAMPLER", "LATENT",)
    FUNCTION = "update"
    CATEGORY = "real-time/sampling"
    DESCRIPTION="Implements batched denoising for faster inference by processing multiple frames in parallel at different denoising steps (StreamDiffusion)"
    
    def __init__(self):
        super().__init__()
        self.frame_buffer_size = None
        self.frame_buffer = None  # Will be a tensor instead of list
        self.x_t_latent_buffer = None
        self.stock_noise = None
        self.cached_noise = {}  # Cache for different expanded versions
        self.debug = False  # Debug flag for print statements
        self.last_batch_shape = None  # Track last batch shape
        # Add CFG state
        self.cfg_type = "self"
        self.guidance_scale = 1.2
        self.delta = 1.0
        # Add cross attention state
        self.prompt = ""
        self.last_prompt = None
        self.kv_cache = None
        self.cross_attention_hook = None
        self.last_model_hash = None
        # Add similarity filter state
        self.similarity_threshold = 0.0
        self.max_skip_frames = 10
        self.similarity_filter = None
        # Add buffer state
        self.num_timesteps = None
        # Add caching for alpha/beta
        self.cached_sigmas = None
        self.cached_alpha_prod_t = None
        self.cached_beta_prod_t = None
        # Add noise sequence and step counter
        self.noise_sequence = None
        self.step_counter = 0
    
    def get_expanded_noise(self, batch_size, scale=1.0):
        """Get expanded noise with caching based on batch size and scale"""
        cache_key = (batch_size, scale)
        if self.stock_noise is None:
            return None
            
        if cache_key not in self.cached_noise:
            self.cached_noise[cache_key] = self.stock_noise.unsqueeze(0).expand(batch_size, -1, -1, -1) * scale
            if self.debug:
                print(f"[StreamBatchSampler] Cached expanded noise for batch_size={batch_size}, scale={scale}")
            
        return self.cached_noise[cache_key]

    def invalidate_noise_cache(self):
        """Clear all cached noise when stock noise changes"""
        self.cached_noise.clear()


    def process_with_cfg(self, model, x, sigma_batch, extra_args):
        """Apply classifier-free guidance based on current cfg_type"""
        # Handle no guidance cases first
        if self.cfg_type == "none" or self.guidance_scale <= 1.0:
            # No guidance - process batch normally
            return model(x, sigma_batch, **extra_args)
            
        # Apply CFG based on type
        if self.cfg_type == "full":
            # Double the batch for cond/uncond
            x_double = torch.cat([x, x], dim=0)
            sigma_double = torch.cat([sigma_batch, sigma_batch], dim=0)
            model_output = model(x_double, sigma_double, **extra_args)
            noise_pred_uncond, noise_pred_text = model_output.chunk(2)
        elif self.cfg_type == "initialize":
            # Add single uncond at start
            x_plus_uc = torch.cat([x[0:1], x], dim=0)
            sigma_plus_uc = torch.cat([sigma_batch[0:1], sigma_batch], dim=0)
            model_output = model(x_plus_uc, sigma_plus_uc, **extra_args)
            noise_pred_uncond = model_output[0:1]
            noise_pred_text = model_output[1:]
            # Update stock noise with uncond prediction
            self.stock_noise = noise_pred_uncond[0].clone()
            self.invalidate_noise_cache()
        else:  # self
            # Use stock noise for uncond
            model_output = model(x, sigma_batch, **extra_args)
            noise_pred_text = model_output 
            
            #TODO:  Eq.7 ϵ_cfg = δϵ̃ + γ(ϵ_c - δϵ̃)
            noise_pred_uncond = self.get_expanded_noise(x.shape[0], scale=self.delta)
            
        # Combine predictions with guidance
        return noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    def setup_cross_attention(self, model, prompt):
        """Set up cross attention optimization for the model"""
        if not prompt:  # Don't use optimization if no prompt provided
            return model

        def optimized_attention_forward(module, x, context=None, mask=None):
            """Optimized cross attention with KV caching and ComfyUI's optimizations"""
            batch_size = x.shape[0]
            
            # Generate query from input
            q = module.to_q(x)
            
            # Check if we need to recompute KV cache
            if self.kv_cache is None or self.last_prompt != prompt:
                if self.debug:
                    print("[StreamBatchSampler] Computing new KV cache")
                k = module.to_k(context)
                v = module.to_v(context)
                self.kv_cache = (k, v)
                self.last_prompt = prompt
            else:
                if self.debug:
                    print("[StreamBatchSampler] Using cached KV pairs")
                k, v = self.kv_cache
            
            # Handle batch size expansion if needed
            if k.shape[0] == 1 and batch_size > 1:
                k = k.expand(batch_size, -1, -1)
                v = v.expand(batch_size, -1, -1)
            
            # Use ComfyUI's optimized attention
            from comfy.ldm.modules.attention import optimized_attention
            out = optimized_attention(q, k, v, module.heads, mask=mask)
            return module.to_out[0](out)

        # Get the actual model from the sampler
        if hasattr(model, "model"):
            unet = model.model
        else:
            return model

        # Only set up hooks if prompt has changed
        if self.last_prompt != prompt:
            # Remove old hooks if they exist
            if self.cross_attention_hook is not None:
                self.cross_attention_hook.remove()
            
            def create_cross_attn_patch(module, input, output):
                """Hook to replace cross attention modules with optimized version"""
                if isinstance(module, torch.nn.Module) and hasattr(module, "to_q"):
                    if not hasattr(module, "_original_forward"):
                        module._original_forward = module.forward
                    module.forward = lambda *args, **kwargs: optimized_attention_forward(module, *args, **kwargs)
                return output

            # Register hooks on cross attention blocks
            def register_hooks(module):
                if isinstance(module, torch.nn.Module) and hasattr(module, "to_q"):
                    self.cross_attention_hook = module.register_forward_hook(create_cross_attn_patch)
            
            unet.apply(register_hooks)
            self.last_prompt = prompt
        
        return model

    def apply_similarity_filter(self, x_0_pred_out):
        """Apply similarity filter to skip similar frames if enabled"""
        if not self.similarity_threshold > 0:
            return x_0_pred_out
            
        if self.similarity_filter is None:
            from .controls.similar_image_filter import SimilarImageFilter
            self.similarity_filter = SimilarImageFilter(
                threshold=self.similarity_threshold,
                max_skip_frame=self.max_skip_frames
            )
        filtered_out = self.similarity_filter(x_0_pred_out)
        if filtered_out is not None:
            return filtered_out
        
        return x_0_pred_out

    def compute_alpha_beta(self, sigmas):
        """Pre-compute alpha and beta terms for the given sigmas"""
        self.cached_sigmas = sigmas.clone()
        # Adjust alpha/beta calculation to use actual timesteps
        self.cached_alpha_prod_t = torch.ones((self.num_timesteps,1,1,1), device=sigmas.device)
        for i in range(self.num_timesteps):
            self.cached_alpha_prod_t[i] = sigmas[i] / sigmas[0]
        self.cached_beta_prod_t = (1 - self.cached_alpha_prod_t)
        if self.debug:
            print("[StreamBatchSampler] Computed new alpha/beta terms")

    def sample(self, model, noise, sigmas, extra_args=None, callback=None, disable=None):
        """Sample with staggered batch denoising steps"""
        extra_args = {} if extra_args is None else extra_args
        
        # Set up cross attention optimization if prompt provided
        if self.prompt:
            model = self.setup_cross_attention(model, self.prompt)
        
        # Get number of frames in batch and available sigmas
        batch_size = noise.shape[0]
        
        # Calculate total batch size based on frame_buffer_size and number of timesteps
        total_batch_size = self.frame_buffer_size * self.num_timesteps
        if batch_size != total_batch_size:
            raise ValueError(f"Batch size ({batch_size}) must match frame_buffer_size * num_timesteps ({total_batch_size})")
        
        # Pre-compute alpha and beta terms if not already computed
        if self.cached_sigmas is None:
            self.compute_alpha_beta(sigmas)
        
        # Replace stock_noise initialization with sequence-based
        if self.stock_noise is None:
            # Get current step's noise from pre-generated sequence
            current_noise = self.noise_sequence[self.step_counter % self.num_timesteps]
            self.stock_noise = current_noise.clone()
        
        # Update step counter for next frame
        self.step_counter += 1
        
        # Scale noise for each frame based on its sigma - vectorized with caching
        sigma_scaled_noise = self.get_expanded_noise(batch_size)
        x = noise + sigma_scaled_noise * sigmas[:self.num_timesteps].view(-1, 1, 1, 1)
            
        # Initialize frame buffer if needed
        if self.x_t_latent_buffer is None and self.num_timesteps > 1:
            self.x_t_latent_buffer = x[0].clone()  # Initialize with noised first frame
            
        # Use buffer for first frame to maintain temporal consistency
        if self.num_timesteps > 1:
            x = torch.cat([self.x_t_latent_buffer.unsqueeze(0), x[1:]], dim=0)
            
        # Run model on entire batch at once
        with torch.no_grad():
            # Process all frames in parallel
            sigma_batch = sigmas[:self.num_timesteps]
            # Process with CFG handling inside process_with_cfg
            denoised_batch = self.process_with_cfg(model, x, sigma_batch, extra_args)
            
            # Update buffer with intermediate results
            if self.num_timesteps > 1:
                # Store result from first frame as buffer for next iteration
                self.x_t_latent_buffer = denoised_batch[0].clone()
                # Return result from last frame
                x_0_pred_out = denoised_batch[-1].unsqueeze(0)
            else:
                x_0_pred_out = denoised_batch
                self.x_t_latent_buffer = None
                
            # Apply similarity filter if enabled
            x_0_pred_out = self.apply_similarity_filter(x_0_pred_out)
                
            # Call callback if provided
            if callback is not None:
                callback({'x': x_0_pred_out, 'i': 0, 'sigma': sigmas[0], 'sigma_hat': sigmas[0], 'denoised': x_0_pred_out})
        
        return x_0_pred_out
    
    def buffer(self, latent, frame_buffer_size=1, num_timesteps=2, always_execute=True):
        """Add new frame to buffer and return batch when ready"""
        self.frame_buffer_size = frame_buffer_size
        self.num_timesteps = num_timesteps
        
        # Extract latent tensor from input and remove batch dimension if present
        x = latent["samples"]
        if x.dim() == 4:  # [B,C,H,W]
            x = x.squeeze(0)  # Remove batch dimension -> [C,H,W]
        
        # Initialize or resize frame buffer if needed
        if self.frame_buffer is None or self.frame_buffer.shape[0] != self.frame_buffer_size:
            # Pre-allocate tensor buffer
            self.frame_buffer = torch.zeros((self.frame_buffer_size,) + x.shape, dtype=x.dtype, device=x.device)
            if self.debug:
                print(f"[StreamFrameBuffer] Pre-allocated tensor buffer for {self.frame_buffer_size} frames")
        
        # Rotate frames and add new frame using tensor operations
        if self.frame_buffer_size > 1:
            self.frame_buffer = torch.roll(self.frame_buffer, -1, dims=0)
        self.frame_buffer[-1] = x
        
        # Create batch by repeating frames for each timestep
        batch = self.frame_buffer.repeat_interleave(self.num_timesteps, dim=0)  # [buffer_size*num_timesteps,C,H,W]
        
        # Return as latent dict
        return {"samples": batch}
    
    def update(self, latent, num_timesteps, frame_buffer_size=1, cfg_type="self", guidance_scale=1.2, delta=1.0, prompt="", 
              similarity_threshold=0.0, max_skip_frames=10, always_execute=True):
        """Create sampler with specified settings and process frame"""
        self.frame_buffer_size = frame_buffer_size
        self.cfg_type = cfg_type
        self.guidance_scale = guidance_scale
        self.delta = delta
        self.prompt = prompt
        self.num_timesteps = num_timesteps
        
        # Update similarity filter settings
        self.similarity_threshold = similarity_threshold
        self.max_skip_frames = max_skip_frames
        if self.similarity_filter is not None:
            self.similarity_filter.set_threshold(similarity_threshold)
            self.similarity_filter.set_max_skip_frame(max_skip_frames)
        
        # Create sampler with cusatom sample function
        sampler = comfy.samplers.KSAMPLER(self.sample)
        
        # Process frame
        buffered = self.buffer(latent)
        
        # Add noise sequence initialization
        if self.noise_sequence is None or self.noise_sequence.shape[0] != num_timesteps:
            latent_shape = latent["samples"].shape[1:]  # [C,H,W]
            self.noise_sequence = torch.randn(
                (num_timesteps,) + latent_shape,  # Proper size format
                device=comfy.model_management.get_torch_device(),
                dtype=latent["samples"].dtype
            )
        
        return (sampler, buffered)
        

class StreamScheduler(ControlNodeBase):
    """Implements StreamDiffusion's efficient timestep selection"""
    
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "update"
    CATEGORY = "real-time/sampling"

    def __init__(self):
        super().__init__()
        self.cached_indices = None
        self.last_t_index_list = None

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "model": ("MODEL",),
            "t_index_list": ("STRING", {
                "default": "32,45",
                "tooltip": "Comma-separated list of timesteps to actually use for denoising. For LCM: use '32,45' for img2img or '0,16,32,45' for txt2img. For SDXL Turbo: use '45,49' for 2 steps or '49' for 1 step (Turbo works best with steps near the end of denoising)"
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

    def parse_timesteps(self, t_index_list):
        """Parse and sort timesteps, with caching"""
        if t_index_list == self.last_t_index_list and self.cached_indices is not None:
            return self.cached_indices
            
        try:
            # Parse and sort in reverse order once
            indices = sorted([int(t.strip()) for t in t_index_list.split(",")], reverse=True)
            self.last_t_index_list = t_index_list
            self.cached_indices = indices
            return indices
        except ValueError as e:
            print(f"Error parsing timesteps: {e}. Using default [32,45]")
            self.last_t_index_list = "32,45"
            self.cached_indices = [45, 32]  # Already sorted in reverse
            return self.cached_indices

    def update(self, model, t_index_list="32,45", num_inference_steps=50, always_execute=True):
        # Get model's sampling parameters
        model_sampling = model.get_model_object("model_sampling")
        
        # Get parsed and sorted timesteps
        t_indices = self.parse_timesteps(t_index_list)
            
        # Create full schedule using normal scheduler
        full_sigmas = comfy.samplers.normal_scheduler(model_sampling, num_inference_steps)
        
        # Select only the sigmas at our desired indices
        # No need to sort t_indices here since they're already sorted
        selected_sigmas = []
        for t in t_indices:
            if t < 0 or t >= num_inference_steps:
                print(f"Warning: timestep {t} out of range [0,{num_inference_steps}), skipping")
                continue
            selected_sigmas.append(float(full_sigmas[t]))
            
        # Add final sigma
        selected_sigmas.append(0.0)
        
        # Create tensor directly on target device
        device = comfy.model_management.get_torch_device()
        selected_sigmas = torch.tensor(selected_sigmas, dtype=torch.float32, device=device)
        return (selected_sigmas,)


