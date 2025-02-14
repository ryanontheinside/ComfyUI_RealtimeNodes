import torch
from .base.control_base import ControlNodeBase
import comfy.model_management
import comfy.samplers
import random
import math


class StreamBatchSampler(ControlNodeBase):
    """Implements batched denoising for faster inference by processing multiple frames in parallel at different denoising steps"""
    
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "update"
    CATEGORY = "real-time/sampling"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "num_steps": ("INT", {
                "default": 4,
                "min": 1,
                "max": 10,
                "step": 1,
                "tooltip": "Number of denoising steps. Should match the frame buffer size."
            }),
            "cfg_type": (["none", "full", "self", "initialize"], {
                "default": "self",
                "tooltip": "Type of CFG to use: none (no guidance), full (standard), self (memory efficient), or initialize (memory efficient with initialization)"
            }),
            "guidance_scale": ("FLOAT", {
                "default": 1.2,
                "min": 1.0,
                "max": 20.0,
                "step": 0.1,
                "tooltip": "CFG guidance scale - 1.0 means no guidance"
            }),
            "delta": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 5.0,
                "step": 0.1,
                "tooltip": "Delta parameter for self/initialize CFG types"
            }),
            "prompt": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Text prompt to use for caching. Only recomputes KV cache when this changes."
            }),
        })
        return inputs
    
    def __init__(self):
        super().__init__()
        self.num_steps = None
        self.frame_buffer = []
        self.x_t_latent_buffer = None
        self.stock_noise = None
        # Add CFG state
        self.cfg_type = "self"
        self.guidance_scale = 1.2
        self.delta = 1.0
        # Add cross attention state
        self.last_prompt = None
        self.kv_cache = None
        self.cross_attention_hook = None
        self.last_model_hash = None
    
    def setup_cross_attention(self, model, prompt):
        """Set up cross attention optimization for the model"""
        print(f"[StreamBatchSampler] Setting up cross attention with prompt: {prompt}")
        
        def cross_attention_forward(module, x, context=None, mask=None):
            """Optimized cross attention with KV caching"""
            batch_size = x.shape[0]
            
            # Generate query from input
            q = module.to_q(x)
            
            # Check if we need to recompute KV cache
            if self.kv_cache is None or self.last_prompt != prompt:
                print("[StreamBatchSampler] Computing new KV cache")
                k = module.to_k(context)
                v = module.to_v(context)
                self.kv_cache = (k, v)
                self.last_prompt = prompt
            else:
                print("[StreamBatchSampler] Using cached KV pairs")
                k, v = self.kv_cache
            
            # Handle batch size expansion if needed
            if k.shape[0] == 1 and batch_size > 1:
                k = k.expand(batch_size, -1, -1)
                v = v.expand(batch_size, -1, -1)
            
            # Reshape for attention
            head_dim = q.shape[-1] // module.heads
            q = q.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)
            k = k.view(-1, k.shape[1], module.heads, head_dim).transpose(1, 2)
            v = v.view(-1, v.shape[1], module.heads, head_dim).transpose(1, 2)
            
            # Compute attention
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            if mask is not None:
                scores = scores + mask
            
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            
            # Reshape output
            out = out.transpose(1, 2).reshape(batch_size, -1, module.heads * head_dim)
            return module.to_out[0](out)
        
        def hook_cross_attention(module, input, output):
            """Hook to replace cross attention modules with optimized version"""
            if isinstance(module, torch.nn.Module) and hasattr(module, "to_q"):
                if not hasattr(module, "_original_forward"):
                    module._original_forward = module.forward
                module.forward = lambda *args, **kwargs: cross_attention_forward(module, *args, **kwargs)
            return output
        
        # Only set up hooks if model has changed or prompt has changed
        model_hash = hash(str(model))
        if model_hash != self.last_model_hash or self.last_prompt != prompt:
            print("[StreamBatchSampler] Setting up new cross attention hooks")
            # Remove old hooks if they exist
            if self.cross_attention_hook is not None:
                self.cross_attention_hook.remove()
            
            # Clone model and apply hooks
            m = model.clone()
            
            def register_hooks(module):
                if isinstance(module, torch.nn.Module) and hasattr(module, "to_q"):
                    self.cross_attention_hook = module.register_forward_hook(hook_cross_attention)
            
            m.model.apply(register_hooks)
            self.last_model_hash = model_hash
            return m
        
        return model
    
    def sample(self, model, noise, sigmas, extra_args=None, callback=None, disable=None):
        """Sample with staggered batch denoising steps and CFG"""
        extra_args = {} if extra_args is None else extra_args
        print(f"[StreamBatchSampler] Starting sampling with {len(sigmas)-1} steps")
        print(f"[StreamBatchSampler] CFG type: {self.cfg_type}, scale: {self.guidance_scale}, delta: {self.delta}")
        
        # Set up cross attention optimization
        prompt = extra_args.get("prompt", "")
        model = self.setup_cross_attention(model, prompt)
        
        # Get number of frames in batch and available sigmas
        batch_size = noise.shape[0]
        num_sigmas = len(sigmas) - 1  # Subtract 1 because last sigma is the target (0.0)
        
        print(f"[StreamBatchSampler] Input sigmas: {sigmas}")
        print(f"[StreamBatchSampler] Input noise shape: {noise.shape}, min: {noise.min():.3f}, max: {noise.max():.3f}")
        
        # Verify batch size matches number of timesteps
        if batch_size != num_sigmas:
            raise ValueError(f"Batch size ({batch_size}) must match number of timesteps ({num_sigmas})")
        
        # Pre-compute alpha and beta terms
        alpha_prod_t = (sigmas[:-1] / sigmas[0]).view(-1, 1, 1, 1)  # [B,1,1,1]
        beta_prod_t = (1 - alpha_prod_t)
        
        print(f"[StreamBatchSampler] Alpha values: {alpha_prod_t.view(-1)}")
        print(f"[StreamBatchSampler] Beta values: {beta_prod_t.view(-1)}")
        
        # Initialize stock noise if needed
        if self.stock_noise is None:
            self.stock_noise = torch.randn_like(noise[0])  # Random noise instead of zeros
            print(f"[StreamBatchSampler] Initialized random stock noise with shape: {self.stock_noise.shape}")
            
        # Scale noise for each frame based on its sigma
        scaled_noise = []
        for i in range(batch_size):
            frame_noise = noise[i] + self.stock_noise * sigmas[i]  # Add scaled noise to input
            scaled_noise.append(frame_noise)
        x = torch.stack(scaled_noise, dim=0)
        print(f"[StreamBatchSampler] Scaled noise shape: {x.shape}, min: {x.min():.3f}, max: {x.max():.3f}")
            
        # Initialize frame buffer if needed
        if self.x_t_latent_buffer is None and num_sigmas > 1:
            self.x_t_latent_buffer = x[0].clone()  # Initialize with noised first frame
            print(f"[StreamBatchSampler] Initialized buffer with shape: {self.x_t_latent_buffer.shape}")
            
        # Use buffer for first frame to maintain temporal consistency
        if num_sigmas > 1:
            x = torch.cat([self.x_t_latent_buffer.unsqueeze(0), x[1:]], dim=0)
            print(f"[StreamBatchSampler] Combined with buffer, shape: {x.shape}")
            
        # Run model on entire batch at once
        with torch.no_grad():
            # Handle CFG based on mode
            if self.cfg_type == "none" or self.guidance_scale <= 1.0:
                # No guidance - process batch normally
                model_output = model(x, sigmas[:-1], **extra_args)
                model_pred = model_output[0]
            else:
                # Apply CFG based on type
                if self.cfg_type == "full":
                    # Double the batch for cond/uncond
                    x_double = torch.cat([x, x], dim=0)
                    sigmas_double = torch.cat([sigmas[:-1], sigmas[:-1]], dim=0)
                    model_output = model(x_double, sigmas_double, **extra_args)
                    noise_pred_uncond, noise_pred_text = model_output[0].chunk(2)
                elif self.cfg_type == "initialize":
                    # Add single uncond at start
                    x_plus_uc = torch.cat([x[0:1], x], dim=0)
                    sigmas_plus_uc = torch.cat([sigmas[:-1][0:1], sigmas[:-1]], dim=0)
                    model_output = model(x_plus_uc, sigmas_plus_uc, **extra_args)
                    noise_pred_uncond = model_output[0][0:1]
                    noise_pred_text = model_output[0][1:]
                    # Update stock noise
                    self.stock_noise = torch.cat([noise_pred_uncond, self.stock_noise[1:]], dim=0)
                else:  # self
                    # Use stock noise for uncond
                    model_output = model(x, sigmas[:-1], **extra_args)
                    noise_pred_text = model_output[0]
                    noise_pred_uncond = self.stock_noise * self.delta
                
                # Combine predictions
                model_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            print(f"[StreamBatchSampler] Model prediction shape: {model_pred.shape}")
            
            # Update stock noise for self/initialize modes
            if (self.cfg_type == "self" or self.cfg_type == "initialize") and num_sigmas > 1:
                # Calculate next alpha/beta
                alpha_next = torch.cat([alpha_prod_t[1:], torch.ones_like(alpha_prod_t[0:1])], dim=0)
                beta_next = torch.cat([beta_prod_t[1:], torch.ones_like(beta_prod_t[0:1])], dim=0)
                
                # Update stock noise
                scaled_noise = beta_prod_t * self.stock_noise
                delta_x = (x - scaled_noise) / alpha_prod_t
                delta_x = alpha_next * delta_x
                delta_x = delta_x / beta_next
                
                # Rotate noise
                init_noise = torch.cat([noise[1:], noise[0:1]], dim=0)
                self.stock_noise = init_noise + delta_x
            
            # Compute denoised result
            denoised_batch = (x - beta_prod_t * model_pred) / alpha_prod_t
            print(f"[StreamBatchSampler] Denoised batch shape: {denoised_batch.shape}")
            
            # Update buffer with intermediate results
            if num_sigmas > 1:
                # Store result from first frame as buffer for next iteration
                self.x_t_latent_buffer = denoised_batch[0].clone()
                print(f"[StreamBatchSampler] Updated buffer with shape: {self.x_t_latent_buffer.shape}")
                
                # Return result from last frame
                x_0_pred_out = denoised_batch[-1].unsqueeze(0)
            else:
                x_0_pred_out = denoised_batch
                self.x_t_latent_buffer = None
                
            # Call callback if provided
            if callback is not None:
                callback({'x': x_0_pred_out, 'i': 0, 'sigma': sigmas[0], 'sigma_hat': sigmas[0], 'denoised': denoised_batch[-1:]})
        
        return x_0_pred_out
    
    def update(self, num_steps=4, cfg_type="self", guidance_scale=1.2, delta=1.0, prompt="", always_execute=True):
        """Create sampler with specified settings"""
        self.num_steps = num_steps
        self.cfg_type = cfg_type
        self.guidance_scale = guidance_scale
        self.delta = delta
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
        
        # Parse timestep list
        try:
            t_index_list = [int(t.strip()) for t in t_index_list.split(",")]
        except ValueError as e:
            print(f"Error parsing timesteps: {e}. Using default [32,45]")
            t_index_list = [32, 45]
            
        # Create full schedule using normal scheduler
        full_sigmas = comfy.samplers.normal_scheduler(model_sampling, num_inference_steps)
        
        # Select only the sigmas at our desired indices, but in reverse order
        # This ensures we go from high noise to low noise
        selected_sigmas = []
        for t in sorted(t_index_list, reverse=True):  # Sort in reverse to go from high noise to low
            if t < 0 or t >= num_inference_steps:
                print(f"Warning: timestep {t} out of range [0,{num_inference_steps}), skipping")
                continue
            selected_sigmas.append(float(full_sigmas[t]))
            
        # Add final sigma
        selected_sigmas.append(0.0)
        
        # Convert to tensor and move to appropriate device
        selected_sigmas = torch.FloatTensor(selected_sigmas).to(comfy.model_management.get_torch_device())
        return (selected_sigmas,)


class StreamFrameBuffer(ControlNodeBase):
    """Accumulates frames to enable staggered batch denoising like StreamDiffusion"""
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "update"
    CATEGORY = "real-time/sampling"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "latent": ("LATENT",),
            "buffer_size": ("INT", {
                "default": 4,
                "min": 1,
                "max": 10,
                "step": 1,
                "tooltip": "Number of frames to buffer before starting batch processing. Should match number of denoising steps."
            }),
        })
        return inputs
    
    def __init__(self):
        super().__init__()
        self.frame_buffer = []  # List to store incoming frames
        self.buffer_size = None
        
    def update(self, latent, buffer_size=4, always_execute=True):
        """Add new frame to buffer and return batch when ready"""
        self.buffer_size = buffer_size
        
        # Extract latent tensor from input and remove batch dimension if present
        x = latent["samples"]
        if x.dim() == 4:  # [B,C,H,W]
            x = x.squeeze(0)  # Remove batch dimension -> [C,H,W]
        
        # Add new frame to buffer
        if len(self.frame_buffer) == 0:
            # First frame - initialize buffer with copies
            self.frame_buffer = [x.clone() for _ in range(self.buffer_size)]
            print(f"[StreamFrameBuffer] Initialized buffer with {self.buffer_size} copies of first frame")
        else:
            # Shift frames forward and add new frame
            self.frame_buffer.pop(0)  # Remove oldest frame
            self.frame_buffer.append(x.clone())  # Add new frame
            print(f"[StreamFrameBuffer] Added new frame to buffer")
            
        # Stack frames into batch
        batch = torch.stack(self.frame_buffer, dim=0)  # [B,C,H,W]
        print(f"[StreamFrameBuffer] Created batch with shape: {batch.shape}")
        
        # Return as latent dict
        return ({"samples": batch},)
