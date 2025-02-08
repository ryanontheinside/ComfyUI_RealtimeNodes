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
            "alpha_prod_t_sqrt": None,
            "beta_prod_t_sqrt": None,
            "init_noise": None,
            "sigmas": None,
            "stock_noise": None,  # For RCFG
        })
        
        # Get latent tensor from dict
        samples = latents["samples"]  # Shape: [1, 4, H, W]
        device = samples.device
        
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
            state["t_indices"] = [int(x.strip()) for x in t_index_list.split(",")]
            
            # Get actual timesteps from scheduler
            timesteps = []
            alpha_prod_t_sqrt_list = []
            beta_prod_t_sqrt_list = []
            
            # Convert sigmas to timesteps and calculate alpha/beta
            for t_idx in state["t_indices"]:
                sigma = sigmas[t_idx]
                # Convert sigma to timestep
                timestep = len(sigmas) - 1 - t_idx
                timesteps.append(timestep)
                
                # Calculate alpha/beta from sigma
                alpha = 1.0 / (1.0 + sigma * sigma)
                beta = sigma * sigma / (1.0 + sigma * sigma)
                
                alpha_prod_t_sqrt_list.append(alpha.sqrt())
                beta_prod_t_sqrt_list.append(beta.sqrt())
            
            # Update t_indices to use actual timesteps
            state["t_indices"] = timesteps
            
            # Stack and reshape for broadcasting
            state["alpha_prod_t_sqrt"] = torch.stack(alpha_prod_t_sqrt_list).view(-1, 1, 1, 1).to(device)
            state["beta_prod_t_sqrt"] = torch.stack(beta_prod_t_sqrt_list).view(-1, 1, 1, 1).to(device)
        
        # Calculate total buffer size based on timesteps and frame buffer
        total_buffer_size = len(state["t_indices"]) * frame_buffer_size
        
        # Initialize buffers if needed
        if state["latent_buffer"] is None:
            state["latent_buffer"] = torch.zeros((total_buffer_size, *samples.shape[1:]), 
                                               dtype=samples.dtype,
                                               device=device)
            # Generate fixed noise pattern
            state["init_noise"] = torch.randn(samples.shape, dtype=samples.dtype, device=device)
            state["stock_noise"] = torch.zeros_like(samples)  # For RCFG
            state["current_idx"] = 0
        
        # Update buffer with current frame
        for i in range(len(state["t_indices"])):
            buffer_idx = i * frame_buffer_size + (state["current_idx"] % frame_buffer_size)
            if do_add_noise:
                # Add scaled noise using proper scheduler parameters
                alpha = state["alpha_prod_t_sqrt"][i]
                beta = state["beta_prod_t_sqrt"][i]
                state["latent_buffer"][buffer_idx] = (
                    alpha * samples[0] +  # Scale clean latent
                    beta * state["init_noise"][0]  # Use fixed noise pattern
                )
            else:
                state["latent_buffer"][buffer_idx] = samples[0]
        
        # Create batched input for denoising
        batched = state["latent_buffer"].view(-1, *samples.shape[1:])  # [total_buffer_size, 4, H, W]
        timesteps = torch.tensor(state["t_indices"], dtype=torch.long, device=device)
        
        # Create sampler
        sampler = comfy.samplers.sampler_object(sampler_name)
        
        # Get sigmas for selected timesteps
        timestep_sigmas = state["sigmas"][timesteps]
        
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
                    # Use stock noise for unconditional
                    noise_pred_uncond = state["stock_noise"] * delta
                    # Update stock noise for next iteration
                    if cfg_type == "self":
                        state["stock_noise"] = noise_pred_text.detach()
                    elif cfg_type == "initialize" and t == timesteps[0]:  # First step
                        state["stock_noise"] = noise_pred_text.detach()
                
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
            
        # Convert latent to BHWC format
        samples = samples.permute(0, 2, 3, 1)  # BCHW -> BHWC
        
        # Modified denoising call:
        denoised = comfy.samplers.sample(
            model,
            state["init_noise"].repeat(len(batched), 1, 1, 1),  
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