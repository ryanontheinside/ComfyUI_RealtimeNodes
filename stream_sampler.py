import torch
from .base.control_base import ControlNodeBase
import comfy.model_management
import comfy.samplers
import random


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
