from .base.control_base import ControlNodeBase
import torch
import comfy.model_management
import comfy.samplers

class StreamDiffusionLatentBuffer(ControlNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "model": ("MODEL",),
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
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {
                    "default": "lcm",
                    "tooltip": "Which scheduler to use for noise scheduling"
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Total number of denoising steps"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT", "LATENT", "TIMESTEPS")
    RETURN_NAMES = ("current_latents", "batched_latents", "timesteps")
    FUNCTION = "update"
    CATEGORY = "real-time/latents/streamdiffusion"
    DESCRIPTION = "Buffers and processes latent frames for StreamDiffusion's batch denoising"

    def update(self, model, latents, t_index_list, frame_buffer_size, do_add_noise, scheduler, num_inference_steps, always_execute=True):
        state = self.get_state({
            "latent_buffer": None,  # Shape: [num_timesteps * frame_buffer_size, 4, H, W]
            "current_idx": 0,
            "t_indices": None,
            "alpha_prod_t_sqrt": None,
            "beta_prod_t_sqrt": None,
            "init_noise": None,
            "sigmas": None
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
        
        # Initialize buffer if needed
        if state["latent_buffer"] is None:
            state["latent_buffer"] = torch.zeros((total_buffer_size, *samples.shape[1:]), 
                                               dtype=samples.dtype,
                                               device=device)
            # Initialize noise with proper seed for reproducibility
            state["init_noise"] = torch.randn_like(state["latent_buffer"])
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
                    beta * state["init_noise"][buffer_idx]  # Add scaled noise
                )
            else:
                state["latent_buffer"][buffer_idx] = samples[0]
        
        # Update index for next frame
        state["current_idx"] = (state["current_idx"] + 1) % frame_buffer_size
        self.set_state(state)
        
        # Return current frame, batched buffer and timesteps
        batched = state["latent_buffer"].view(-1, *samples.shape[1:])  # Reshape to [total_buffer_size, 4, H, W]
        
        # Create timesteps tensor
        timesteps = torch.tensor(state["t_indices"], dtype=torch.long, device=device)
        
        return (
            {"samples": samples},  # Current frame [1, 4, H, W]
            {"samples": batched},  # Batched buffer [total_buffer_size, 4, H, W]
            timesteps  # Timesteps tensor [num_timesteps]
        )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return super().IS_CHANGED(**kwargs)


class StreamDiffusionRCFGGuider:
    def __init__(self, model, cfg_type="self", guidance_scale=1.0, delta=1.0):
        self.model = model
        self.cfg_type = cfg_type
        self.guidance_scale = guidance_scale
        self.delta = delta
        self.stock_noise = None
        self.init_noise = None
        
    def __call__(self, x, timestep, cond, uncond, cfg_scale, model_options={}):
        # Initialize stock noise if needed
        if self.stock_noise is None:
            self.stock_noise = torch.zeros_like(x)
            self.init_noise = torch.randn_like(x)
        
        with torch.no_grad():
            # Get conditional prediction
            noise_pred_text = self.model.apply_model(x, timestep, cond)
            
            # Handle different CFG types
            if self.cfg_type == "none" or cfg_scale <= 1.0:
                return noise_pred_text
            
            if self.cfg_type == "full":
                # Get unconditional prediction
                noise_pred_uncond = self.model.apply_model(x, timestep, uncond)
            elif self.cfg_type == "self" or self.cfg_type == "initialize":
                # Use stock noise for unconditional
                noise_pred_uncond = self.stock_noise * self.delta
                # Update stock noise for next iteration
                if self.cfg_type == "self":
                    self.stock_noise = noise_pred_text.detach()
                elif self.cfg_type == "initialize" and timestep == timestep[0]:  # First step
                    self.stock_noise = noise_pred_text.detach()
            
            # Apply CFG
            return noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

class StreamDiffusionRCFG(ControlNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latents": ("LATENT",),
                "timesteps": ("TIMESTEPS",),
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
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "update"
    CATEGORY = "real-time/sampling/streamdiffusion"
    DESCRIPTION = "Applies StreamDiffusion's Residual CFG optimizations to noise predictions"

    def update(self, model, positive, negative, latents, timesteps, cfg_type, guidance_scale, delta, sampler_name, scheduler, steps, always_execute=True):
        # Create RCFG guider
        rcfg_guider = StreamDiffusionRCFGGuider(model.model, cfg_type, guidance_scale, delta)
        
        # Create sampler
        sampler = comfy.samplers.sampler_object(sampler_name)
        
        # Calculate sigmas
        sigmas = comfy.samplers.calculate_sigmas(
            model.model.model_sampling,
            scheduler,
            steps
        ).to(model.model.device)
        
        # Get sigmas for selected timesteps
        timestep_sigmas = sigmas[timesteps]
        
        # Sample with RCFG
        samples = comfy.samplers.sample(
            model,
            latents["samples"],  # Use input as noise
            positive, 
            negative,
            guidance_scale,
            model.model.device,
            sampler,
            timestep_sigmas,
            model_options={"custom_cfg": rcfg_guider}  # Use our custom CFG
        )
        
        return ({"samples": samples},)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return super().IS_CHANGED(**kwargs)


class StreamDiffusionSampler(ControlNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latents": ("LATENT",),
                "timesteps": ("TIMESTEPS",),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale"
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "update"
    CATEGORY = "real-time/sampling"
    DESCRIPTION = "StreamDiffusion's optimized sampler for real-time generation"

    def update(self, model, positive, negative, latents, timesteps, sampler_name, cfg, num_inference_steps, seed, always_execute=True):
        state = self.get_state({
            "generator": None
        })
        
        # Get input latents
        x_t = latents["samples"]  # Shape: [batch_size, 4, H, W]
        
        # Initialize generator if needed
        if state["generator"] is None:
            state["generator"] = torch.Generator(device=model.model.device)
            state["generator"].manual_seed(seed)
        
        # Get sampler object
        sampler = comfy.samplers.sampler_object(sampler_name)
        
        # Calculate sigmas for timesteps
        sigmas = comfy.samplers.calculate_sigmas(
            model.model.model_sampling,
            "exponential",  # Use exponential scheduler by default
            num_inference_steps
        ).to(model.model.device)
        
        # Get sigmas for selected timesteps
        timestep_sigmas = sigmas[timesteps]
        
        # Sample using ComfyUI's infrastructure
        samples = comfy.samplers.sample(
            model, 
            x_t,  # Use input latents as noise
            positive, 
            negative,
            cfg,
            model.model.device,
            sampler,
            timestep_sigmas,
            denoise_mask=None,  # No mask support yet
            seed=seed
        )
        
        self.set_state(state)
        
        return ({"samples": samples},)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return super().IS_CHANGED(**kwargs)