import torch
import comfy.model_management
import comfy.samplers
import random
import time
import os
import json
from datetime import datetime
from functools import wraps
import cProfile
import pstats
import io

# Simple profiling setup - SINGLE FILE
PROFILE_FILE = "stream_sampler_profile.json"
PROFILE_DATA = []

def profile_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start timing
        start_time = time.time()
        
        # Check memory before
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / (1024 * 1024)
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # End timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        # Check memory after
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / (1024 * 1024)
            mem_diff = mem_after - mem_before
        else:
            mem_diff = 0
        
        exec_time = end_time - start_time
        
        # Get shape info if available
        shape_info = "unknown"
        if len(args) > 1 and hasattr(args[1], 'shape'):  # If this is sample(), args[1] is noise
            shape_info = str(args[1].shape)
        
        # Log results
        entry = {
            'function': func.__name__,
            'execution_time': exec_time,
            'memory_diff_mb': mem_diff,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'shape': shape_info
        }
        
        PROFILE_DATA.append(entry)
        
        # Print to console
        print(f"PROFILE: {func.__name__} - Time: {exec_time:.4f}s, Memory: {mem_diff:.2f}MB")
        
        # Save to file after each run
        save_profile_data()
            
        return result
    return wrapper


def profile_cprofile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create profiler
        pr = cProfile.Profile()
        pr.enable()
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Disable profiler
        pr.disable()
        
        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        
        # Add to the most recent profile entry
        if PROFILE_DATA:
            PROFILE_DATA[-1]['cprofile_data'] = s.getvalue()
        
        return result
    return wrapper


def save_profile_data():
    """Save all profiling data to a single JSON file"""
    if not PROFILE_DATA:
        return
    
    # Convert to serializable format
    json_data = []
    for entry in PROFILE_DATA:
        serializable_entry = {
            'function': entry['function'],
            'execution_time': float(entry['execution_time']),
            'memory_diff_mb': float(entry['memory_diff_mb']),
            'timestamp': entry['timestamp'],
            'shape': entry.get('shape', 'unknown')
        }
        
        if 'cprofile_data' in entry:
            # Only store recent cprofile data to keep file size manageable
            if len(json_data) < 10 or len(json_data) % 10 == 0:
                serializable_entry['cprofile_data'] = entry['cprofile_data']
        
        json_data.append(serializable_entry)
    
    # Write to a single file
    with open(PROFILE_FILE, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Calculate stats
    sample_times = [x['execution_time'] for x in PROFILE_DATA if x['function'] == 'sample']
    if sample_times:
        avg_time = sum(sample_times) / len(sample_times)
        min_time = min(sample_times)
        max_time = max(sample_times)
        print(f"PROFILE SUMMARY: {len(sample_times)} runs, Avg: {avg_time:.4f}s, Min: {min_time:.4f}s, Max: {max_time:.4f}s")
    
    print(f"Profiling data saved to {PROFILE_FILE}")


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
    
    @profile_time
    @profile_cprofile
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
    
    @profile_time
    def update(self, num_steps=4):
        """Create sampler with specified settings"""
        self.num_steps = num_steps
        sampler = comfy.samplers.KSAMPLER(self.sample)
        return (sampler,)

# Print setup info when module is imported
print(f"StreamBatchSampler profiling enabled. Results will be saved to {PROFILE_FILE}")


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