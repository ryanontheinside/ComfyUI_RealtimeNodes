import torch
import comfy.model_management
import comfy.samplers
import random
import time
import os

class StreamBatchSampler:
    
    
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "update"
    CATEGORY = "StreamPack/sampling"
    DESCRIPTION = "Implements batched denoising for faster inference by processing multiple frames in parallel at different denoising steps. Also adds temportal consistency to the denoising process."
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_steps": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of denoising steps. Should match the frame buffer size."
                }),
            },
        }
    
    def __init__(self):
        self.num_steps = None
        self.frame_buffer = []
        self.x_t_latent_buffer = None
        self.stock_noise = None
        self.is_txt2img_mode = False
        
        # Initialize all optimization buffers as None
        self.zeros_reference = None
        self.random_noise_buffer = None
        self.sigmas_view_buffer = None
        self.expanded_stock_noise = None
        self.working_buffer = None
        self.output_buffer = None
    
    def sample(self, model, noise, sigmas, extra_args=None, callback=None, disable=None):
        """Sample with staggered batch denoising steps - Optimized version"""
        extra_args = {} if extra_args is None else extra_args
        
        # Get number of frames in batch and available sigmas
        batch_size = noise.shape[0]
        num_sigmas = len(sigmas) - 1  # Subtract 1 because last sigma is the target (0.0)
        
        # Optimization 1: Reuse zeros buffer for txt2img detection
        if self.zeros_reference is None:
            # We only need a small reference tensor for comparison, not a full tensor
            self.zeros_reference = torch.zeros(1, device=noise.device, dtype=noise.dtype)
        
        # Check if noise tensor is all zeros - functionally identical but more efficient
        self.is_txt2img_mode = torch.abs(noise).sum() < 1e-5
        
        # Noise handling with memory optimization
        if self.is_txt2img_mode:
            # Optimization 2: If txt2img mode, reuse the noise tensor directly
            # instead of allocating new memory
            if self.random_noise_buffer is None or self.random_noise_buffer.shape != noise.shape:
                self.random_noise_buffer = torch.empty_like(noise)
            
            # Generate random noise in-place
            self.random_noise_buffer.normal_()
            x = self.random_noise_buffer  # Use pre-allocated buffer
        else:
            # If not txt2img, we'll still need to add noise later
            x = noise  # No need to copy, will add noise later
        
        # Verify batch size matches number of timesteps
        if batch_size != num_sigmas:
            raise ValueError(f"Batch size ({batch_size}) must match number of timesteps ({num_sigmas})")
        
        # Pre-compute alpha and beta terms
        alpha_prod_t = (sigmas[:-1] / sigmas[0]).view(-1, 1, 1, 1)  # [B,1,1,1]
        beta_prod_t = (1 - alpha_prod_t)
        
        # Optimization 3: Initialize stock noise with reuse
        if self.stock_noise is None or self.stock_noise.shape != noise[0].shape:
            self.stock_noise = torch.empty_like(noise[0])
            self.stock_noise.normal_()  # Generate random noise in-place
        
        # Optimization 4: Pre-allocate and reuse view buffer for sigmas
        if self.sigmas_view_buffer is None or self.sigmas_view_buffer.shape[0] != len(sigmas)-1:
            self.sigmas_view_buffer = torch.empty((len(sigmas)-1, 1, 1, 1), 
                                               device=sigmas.device, 
                                               dtype=sigmas.dtype)
        # In-place copy of sigmas view
        self.sigmas_view_buffer.copy_(sigmas[:-1].view(-1, 1, 1, 1))
        
        # Optimization 5: Eliminate unsqueeze allocation by pre-expanding stock noise
        if self.expanded_stock_noise is None or self.expanded_stock_noise.shape[0] != batch_size:
            self.expanded_stock_noise = self.stock_noise.expand(batch_size, *self.stock_noise.shape)
        
        # Apply noise with pre-allocated buffers - no new memory allocation
        if not self.is_txt2img_mode:  # Already handled txt2img case above
            # If we need a working buffer separate from noise input:
            if id(x) == id(noise):  # They're the same object, need a separate buffer
                if self.working_buffer is None or self.working_buffer.shape != noise.shape:
                    self.working_buffer = torch.empty_like(noise)
                x = self.working_buffer
                # Add noise to input
                torch.add(noise, self.expanded_stock_noise * self.sigmas_view_buffer, out=x)
        
        # Initialize and manage latent buffer with memory optimization
        if (self.x_t_latent_buffer is None or self.is_txt2img_mode) and num_sigmas > 1:
            # Optimization 6: Pre-allocate or resize as needed
            if self.x_t_latent_buffer is None or self.x_t_latent_buffer.shape != x[0].shape:
                self.x_t_latent_buffer = torch.empty_like(x[0])
            # In-place copy instead of clone
            self.x_t_latent_buffer.copy_(x[0])
        
        # Use buffer for first frame to maintain temporal consistency
        if num_sigmas > 1:
            # In-place update - no new allocation
            x[0].copy_(self.x_t_latent_buffer)
        
        # Run model on entire batch at once
        with torch.no_grad():
            # Process all frames in parallel
            sigma_batch = sigmas[:-1]
            
            denoised_batch = model(x, sigma_batch, **extra_args)
            
            # Update buffer with intermediate results
            if num_sigmas > 1:
                # Store result from first frame as buffer for next iteration
                self.x_t_latent_buffer.copy_(denoised_batch[0])  # In-place update
                
                # Optimization 7: Pre-allocate output buffer
                if self.output_buffer is None or self.output_buffer.shape != (1, *denoised_batch[-1].shape):
                    self.output_buffer = torch.empty(1, *denoised_batch[-1].shape, 
                                                  device=denoised_batch.device,
                                                  dtype=denoised_batch.dtype)
                # Copy the result directly to pre-allocated buffer
                self.output_buffer[0].copy_(denoised_batch[-1])
                x_0_pred_out = self.output_buffer
            else:
                x_0_pred_out = denoised_batch
                self.x_t_latent_buffer = None
            
            # Call callback if provided
            if callback is not None:
                callback({'x': x_0_pred_out, 'i': 0, 'sigma': sigmas[0], 'sigma_hat': sigmas[0], 'denoised': denoised_batch[-1:]})
        
        return x_0_pred_out
    
    def update(self, num_steps=4):
        """Create sampler with specified settings"""
        self.num_steps = num_steps
        sampler = comfy.samplers.KSAMPLER(self.sample)
        return (sampler,)


class StreamScheduler:
    
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "update"
    CATEGORY = "StreamPack/sampling"
    DESCRIPTION = "Implements StreamDiffusion's efficient timestep selection. Use in conjunction with StreamBatchSampler."
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
            },
        }

    def update(self, model, t_index_list="32,45", num_inference_steps=50):
        # Get model's sampling parameters
        model_sampling = model.get_model_object("model_sampling")
        
        # Parse timestep list
        try:
            t_index_list = [int(t.strip()) for t in t_index_list.split(",")]
        except ValueError as e:
            
            t_index_list = [32, 45]
            
        # Create full schedule using normal scheduler
        full_sigmas = comfy.samplers.normal_scheduler(model_sampling, num_inference_steps)
        
        # Select only the sigmas at our desired indices, but in reverse order
        # This ensures we go from high noise to low noise
        selected_sigmas = []
        for t in sorted(t_index_list, reverse=True):  # Sort in reverse to go from high noise to low
            if t < 0 or t >= num_inference_steps:
                
                continue
            selected_sigmas.append(float(full_sigmas[t]))
            
        # Add final sigma
        selected_sigmas.append(0.0)
        
        # Convert to tensor and move to appropriate device
        selected_sigmas = torch.FloatTensor(selected_sigmas).to(comfy.model_management.get_torch_device())
        return (selected_sigmas,)


class StreamFrameBuffer:
    
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "update"
    CATEGORY = "StreamPack/sampling"
    DESCRIPTION = "Accumulates frames to enable staggered batch denoising like StreamDiffusion. Use in conjunction with StreamBatchSampler"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "buffer_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of frames to buffer before starting batch processing. Should match number of denoising steps."
                }),
            },
        }
    
    def __init__(self):
        self.frame_buffer = None  # Tensor of shape [buffer_size, C, H, W]
        self.buffer_size = None
        self.buffer_pos = 0  # Current position in ring buffer
        self.is_initialized = False  # Track buffer initialization
        self.is_txt2img_mode = False
    
    def update(self, latent, buffer_size=4):
        """Add new frame to buffer and return batch when ready"""
        self.buffer_size = buffer_size
        
        # Extract latent tensor from input and remove batch dimension if present
        x = latent["samples"]
        
        # Check if this is an empty latent (for txt2img)
        is_empty_latent = x.numel() == 0 or (x.dim() > 0 and x.shape[0] == 0)
        
        if is_empty_latent:
            self.is_txt2img_mode = True

            # Create empty latents with correct dimensions for txt2img
            # Get dimensions from latent dict
            height = latent.get("height", 512)
            width = latent.get("width", 512)
            
            # Calculate latent dimensions (typically 1/8 of image dimensions for SD)
            latent_height = height // 8
            latent_width = width // 8
            
            # Create zero tensor with correct shape
            x = torch.zeros((4, latent_height, latent_width), 
                           device=comfy.model_management.get_torch_device())

        elif x.dim() == 4:  # [B,C,H,W]
            self.is_txt2img_mode = False
            x = x.squeeze(0)  # Remove batch dimension -> [C,H,W]
        
        # Optimization: Initialize or resize frame_buffer as a tensor
        if not self.is_initialized or self.frame_buffer.shape[0] != self.buffer_size or \
           self.frame_buffer.shape[1:] != x.shape:
            # Pre-allocate buffer with correct shape
            self.frame_buffer = torch.zeros(
                (self.buffer_size, *x.shape),
                device=x.device,
                dtype=x.dtype
            )
            if self.is_txt2img_mode or not self.is_initialized:
                # Optimization: Use broadcasting to fill buffer with copies
                self.frame_buffer[:] = x.unsqueeze(0)  # Broadcast x to [buffer_size, C, H, W]

            self.is_initialized = True
            self.buffer_pos = 0
        else:
            # Add new frame to buffer using ring buffer logic
            self.frame_buffer[self.buffer_pos] = x  # In-place update
            self.buffer_pos = (self.buffer_pos + 1) % self.buffer_size  # Circular increment
        
        # Optimization: frame_buffer is already a tensor batch, no need to stack
        batch = self.frame_buffer

        
        # Return as latent dict with preserved dimensions
        result = {"samples": batch}
        
        # Preserve height and width if present in input
        if "height" in latent:
            result["height"] = latent["height"]
        if "width" in latent:
            result["width"] = latent["width"]
            
        return (result,)