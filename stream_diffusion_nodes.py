from .base.control_base import ControlNodeBase
import torch
import comfy.model_management
import comfy.samplers
from .stream_attention import StreamCrossAttention, UNetBatchedStream


class StreamDiffusionPipeline(ControlNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latents": ("LATENT",),
                "t_index_list": ("STRING", {
                    "default": "0,16,32,45",
                    "tooltip": "Comma-separated list of timestep indices"
                }),
                "frame_buffer_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Number of frames to buffer"
                }),
                "do_add_noise": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to add noise to intermediate states"
                }),
                "cfg_type": (["none", "full", "self", "initialize"],),
                "guidance_scale": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "How strong the guidance should be"
                }),
                "delta": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Scaling factor for residual noise"
                }),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {
                    "default": "lcm",
                    "tooltip": "Which scheduler to use for noise scheduling"
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Total number of denoising steps"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "update"
    CATEGORY = "real-time/streamdiffusion"
    DESCRIPTION = "StreamDiffusion pipeline with batched denoising and RCFG"

    def update(self, model, positive, negative, latents, t_index_list, frame_buffer_size, do_add_noise, 
              cfg_type, guidance_scale, delta, scheduler, sampler_name, num_inference_steps, always_execute=True):
        state = self.get_state({
            "latent_buffer": None,  # Shape: [num_timesteps * frame_buffer_size, 4, H, W]
            "current_idx": 0,
            "t_indices": None,
            "sigmas": None,
            "stock_noise": None,  # For RCFG
            "timesteps": None,
            "current_buffer_size": None,  # Track buffer size
        })
        
        # Get latent tensor from dict
        samples = latents["samples"]  # Shape: [1, 4, H, W]
        device = samples.device
        
        # Check if buffer size changed
        if state["current_buffer_size"] != frame_buffer_size:
            state["latent_buffer"] = None  # Force reinitialization
            state["current_buffer_size"] = frame_buffer_size
            state["current_idx"] = 0
        
        # Initialize scheduler and timesteps if needed
        if state["sigmas"] is None:
            # Calculate sigmas using ComfyUI's scheduler system
            sigmas = comfy.samplers.calculate_sigmas(
                model.model.model_sampling,
                scheduler,
                num_inference_steps
            ).to(device)
            state["sigmas"] = sigmas
            
            # Parse t_index_list and get timesteps
            t_indices = [int(x.strip()) for x in t_index_list.split(",")]
            
            # Get actual timesteps from scheduler
            timesteps = []
            for t_idx in t_indices:
                timesteps.append(len(sigmas) - 1 - t_idx)
            
            # Store timesteps for later use
            state["timesteps"] = torch.tensor(timesteps, dtype=torch.long, device=device)
            state["t_indices"] = torch.tensor(t_indices, dtype=torch.long, device=device)
        
        # Calculate total buffer size based on timesteps and frame buffer
        total_buffer_size = len(state["timesteps"]) * frame_buffer_size
        
        # Initialize buffers if needed
        if state["latent_buffer"] is None:
            state["latent_buffer"] = torch.zeros((total_buffer_size, *samples.shape[1:]), 
                                               dtype=samples.dtype,
                                               device=device)
            # Initialize stock noise for RCFG
            state["stock_noise"] = torch.zeros_like(samples)
            state["current_idx"] = 0
        
        # Update buffer with current frame
        for i in range(len(state["timesteps"])):
            buffer_idx = i * frame_buffer_size + (state["current_idx"] % frame_buffer_size)
            if do_add_noise:
                # Get sigma for current timestep
                sigma = state["sigmas"][state["t_indices"][i]]
                # Add noise using sigma scheduling
                noise = torch.randn_like(samples)
                state["latent_buffer"][buffer_idx] = samples[0] + noise[0] * sigma
            else:
                state["latent_buffer"][buffer_idx] = samples[0]
        
        # Create batched input for denoising
        batched = state["latent_buffer"]  # Already in right shape [total_buffer_size, 4, H, W]
        
        # Get sigmas for selected timesteps
        timestep_sigmas = state["sigmas"][state["t_indices"]]
        
        # Create sampler
        sampler = comfy.samplers.sampler_object(sampler_name)
        
        # Create custom CFG function for the sampler
        def custom_cfg(x, t, cond, uncond, cfg_scale, model_options={}):
            with torch.no_grad():
                # Get conditional prediction
                noise_pred_text = model.model.apply_model(x, t, cond)
                
                # Handle different CFG types
                if cfg_type == "none" or cfg_scale <= 1.0:
                    return noise_pred_text
                
                if cfg_type == "full":
                    # Get unconditional prediction
                    noise_pred_uncond = model.model.apply_model(x, t, uncond)
                elif cfg_type == "self" or cfg_type == "initialize":
                    # Use stock noise for unconditional with delta scaling
                    noise_pred_uncond = state["stock_noise"].repeat(x.shape[0], 1, 1, 1) * delta
                    # Update stock noise for next iteration
                    if cfg_type == "self":
                        # Update per batch element
                        state["stock_noise"] = noise_pred_text.reshape(-1, frame_buffer_size, *noise_pred_text.shape[1:])[-1]
                    elif cfg_type == "initialize" and t == timestep_sigmas[0]:  # First step
                        state["stock_noise"] = noise_pred_text[-1:]  # Use last prediction
                
                # Apply CFG
                return noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
        
        # Patch the attention layers first time
        if not hasattr(self, "_attention_patched"):
            StreamCrossAttention.initialize_attention()
            self._attention_patched = True
            
        # Wrap model with batched stream handler
        if not hasattr(model, "_stream_wrapped"):
            model._stream_wrapped = UNetBatchedStream(model)
            model.set_model_unet_function_wrapper(model._stream_wrapped)
        
        # Modified denoising call:
        denoised = comfy.samplers.sample(
            model,
            torch.randn_like(batched),  # Fresh noise per batch
            positive,
            negative,
            guidance_scale,
            model.model.device,
            sampler,
            timestep_sigmas,
            model_options={
                "custom_cfg": custom_cfg,
                "stream_batch": True  # Enable batched mode
            },
            latent_image=batched,
            denoise_mask=None,
            seed=None
        )
        
        # Update index for next frame
        state["current_idx"] = (state["current_idx"] + 1) % frame_buffer_size
        self.set_state(state)
        
        # Return denoised result
        return ({"samples": denoised[-1:]},)  # Return last frame

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return super().IS_CHANGED(**kwargs)

    def terminate(self):
        """Clean up when node is deleted"""
        if hasattr(self, "_attention_patched"):
            # Restore original attention implementation
            from comfy.ldm.modules.attention import CrossAttention
            CrossAttention.forward = StreamCrossAttention.original_attention