import torch
from .base.control_base import ControlNodeBase
import comfy.model_management
import comfy.samplers
import random


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
        })
        return inputs
    
    def __init__(self):
        super().__init__()
        self.num_steps = None
        self.frame_buffer = []
        self.x_t_latent_buffer = None
        self.stock_noise = None
        self.is_txt2img_mode = False
    
    def sample(self, model, noise, sigmas, extra_args=None, callback=None, disable=None):
        """Sample with staggered batch denoising steps"""
        extra_args = {} if extra_args is None else extra_args
        
        # Get number of frames in batch and available sigmas
        batch_size = noise.shape[0]
        num_sigmas = len(sigmas) - 1  # Subtract 1 because last sigma is the target (0.0)
        
        # Detect if we're in text-to-image mode by checking if noise is all zeros
        # This happens when empty latents are provided
        self.is_txt2img_mode = torch.allclose(noise, torch.zeros_like(noise), atol=1e-6)
        
        if self.is_txt2img_mode:
            # For text-to-image, we'll use pure random noise
            noise = torch.randn_like(noise)
        
        # Verify batch size matches number of timesteps
        if batch_size != num_sigmas:
            raise ValueError(f"Batch size ({batch_size}) must match number of timesteps ({num_sigmas})")
        
        # Pre-compute alpha and beta terms
        alpha_prod_t = (sigmas[:-1] / sigmas[0]).view(-1, 1, 1, 1)  # [B,1,1,1]
        beta_prod_t = (1 - alpha_prod_t)
        
        
        # Initialize stock noise if needed
        if self.stock_noise is None or self.is_txt2img_mode:  # Kept original condition for functional equivalence
            self.stock_noise = torch.randn_like(noise[0])  # Random noise instead of zeros
        
        # Optimization: Vectorize noise scaling instead of looping
        sigmas_view = sigmas[:-1].view(-1, 1, 1, 1)  # Reshape for broadcasting
        if self.is_txt2img_mode:
            x = self.stock_noise.unsqueeze(0) * sigmas_view  # Broadcast stock_noise across batch
        else:
            x = noise + self.stock_noise.unsqueeze(0) * sigmas_view  # Add scaled noise to input
            
        # Initialize frame buffer if needed
        if (self.x_t_latent_buffer is None or self.is_txt2img_mode) and num_sigmas > 1:  # Kept original condition
            # Optimization: Pre-allocate and copy instead of clone
            self.x_t_latent_buffer = torch.empty_like(x[0])  # Pre-allocate memory
            self.x_t_latent_buffer.copy_(x[0])  # In-place copy
            
        # Use buffer for first frame to maintain temporal consistency
        if num_sigmas > 1:
            # Optimization: Update in-place instead of concatenating
            x[0] = self.x_t_latent_buffer  # Replace first frame with buffer
            
        # Run model on entire batch at once
        with torch.no_grad():
            # Process all frames in parallel
            sigma_batch = sigmas[:-1]
            
            denoised_batch = model(x, sigma_batch, **extra_args)
            
            # Update buffer with intermediate results
            if num_sigmas > 1:
                # Store result from first frame as buffer for next iteration
                # Optimization: Use in-place copy instead of clone
                self.x_t_latent_buffer.copy_(denoised_batch[0])  # Update buffer in-place
                
                # Return result from last frame
                x_0_pred_out = denoised_batch[-1].unsqueeze(0)
            else:
                x_0_pred_out = denoised_batch
                self.x_t_latent_buffer = None
                
            # Call callback if provided
            if callback is not None:
                callback({'x': x_0_pred_out, 'i': 0, 'sigma': sigmas[0], 'sigma_hat': sigmas[0], 'denoised': denoised_batch[-1:]})
        
        return x_0_pred_out
    
    def update(self, num_steps=4, always_execute=True):
        """Create sampler with specified settings"""
        self.num_steps = num_steps
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
        # Optimization: Replace list with a pre-allocated tensor ring buffer
        self.frame_buffer = None  # Tensor of shape [buffer_size, C, H, W]
        self.buffer_size = None
        self.buffer_pos = 0  # Current position in ring buffer
        self.is_initialized = False  # Track buffer initialization
        self.is_txt2img_mode = False
    
    def update(self, latent, buffer_size=4, always_execute=True):
        """Add new frame to buffer and return batch when ready"""
        self.buffer_size = buffer_size
        
        # Extract latent tensor from input and remove batch dimension if present
        x = latent["samples"]
        
        # Check if this is an empty latent (for txt2img)
        is_empty_latent = x.numel() == 0 or (x.dim() > 0 and x.shape[0] == 0)
        
        if is_empty_latent:
            self.is_txt2img_mode = True
            print(f"[StreamFrameBuffer] Detected empty latent for text-to-image mode")
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
            print(f"[StreamFrameBuffer] Created empty latent with shape: {x.shape}")
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
                print(f"[StreamFrameBuffer] Initialized buffer with {self.buffer_size} copies of frame")
            self.is_initialized = True
            self.buffer_pos = 0
        else:
            # Add new frame to buffer using ring buffer logic
            self.frame_buffer[self.buffer_pos] = x  # In-place update
            print(f"[StreamFrameBuffer] Added new frame to buffer at position {self.buffer_pos}")
            self.buffer_pos = (self.buffer_pos + 1) % self.buffer_size  # Circular increment
        
        # Optimization: frame_buffer is already a tensor batch, no need to stack
        batch = self.frame_buffer
        print(f"[StreamFrameBuffer] Created batch with shape: {batch.shape}")
        
        # Return as latent dict with preserved dimensions
        result = {"samples": batch}
        
        # Preserve height and width if present in input
        if "height" in latent:
            result["height"] = latent["height"]
        if "width" in latent:
            result["width"] = latent["width"]
            
        return (result,)