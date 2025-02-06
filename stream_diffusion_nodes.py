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
                "latents": ("LATENT",),
                "buffer_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Number of latent frames to buffer"
                }),
                "add_noise": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to add noise to intermediate states"
                }),
                "noise_alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Alpha for noise scaling"
                }),
                "noise_beta": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Beta for noise scaling"
                })
            }
        }
    
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("current_latents", "batched_latents")
    FUNCTION = "update"
    CATEGORY = "real-time/latents/streamdiffusion"
    DESCRIPTION = "Buffers and processes latent frames for StreamDiffusion's batch denoising"

    def update(self, latents, buffer_size, add_noise, noise_alpha, noise_beta, always_execute=True):
        state = self.get_state({
            "latent_buffer": None,
            "current_idx": 0
        })
        
        # Get latent tensor from dict
        samples = latents["samples"]  # Shape: [1, 4, H, W]
        
        # Initialize buffer if needed
        if state["latent_buffer"] is None:
            state["latent_buffer"] = torch.zeros((buffer_size, *samples.shape[1:]), 
                                               dtype=samples.dtype,
                                               device=samples.device)
            state["current_idx"] = 0
        
        # Update buffer with current frame
        state["latent_buffer"][state["current_idx"]] = samples[0].detach()  # Remove batch dim
        
        # Create batched latents for processing
        if add_noise:
            # Add scaled noise to previous frames like StreamDiffusion
            noise = torch.randn_like(state["latent_buffer"])
            batched = noise_alpha * state["latent_buffer"] + noise_beta * noise
        else:
            batched = state["latent_buffer"].clone()
            
        # Update index for next frame
        state["current_idx"] = (state["current_idx"] + 1) % buffer_size
        self.set_state(state)
        
        # Return current frame and batched buffer
        # Both in ComfyUI's LATENT format with batch dimension
        # Reshape batched to [batch_size, channels, H, W] where batch_size = buffer_size
        batched = batched.view(-1, *samples.shape[1:])  # Flatten to [buffer_size, 4, H, W]
        
        return (
            {"samples": samples},  # Current frame [1, 4, H, W]
            {"samples": batched}  # Batched buffer [buffer_size, 4, H, W]
        )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return super().IS_CHANGED(**kwargs)


class StreamDiffusionRCFG(ControlNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "rcfg_type": (["self", "initialize"],),
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
                })
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive_cond", "negative_cond")
    FUNCTION = "update"
    CATEGORY = "real-time/conditioning/streamdiffusion"
    DESCRIPTION = "Applies StreamDiffusion's Residual CFG optimizations to conditioning"

    def __init__(self):
        super().__init__()  # Initialize ControlNodeBase
        
    def update(self, positive, negative, rcfg_type, guidance_scale, delta, always_execute=True):
        state = self.get_state({
            "stock_noise": None,
            "timestep_count": 0
        })
        
        # Get first conditioning tuple from input lists
        pos_cond, pos_meta = positive[0]  # Each conditioning is a list of tuples
        neg_cond, neg_meta = negative[0]
        
        # Initialize stock_noise if needed
        if state["stock_noise"] is None:
            if rcfg_type == "self":
                state["stock_noise"] = torch.zeros_like(neg_cond)
            elif rcfg_type == "initialize" and state["timestep_count"] == 0:
                state["stock_noise"] = neg_cond.clone()  # Store initial negative cond
        
        # Calculate residual negative conditioning
        residual_neg_cond = state["stock_noise"] * delta
        
        # Update stock noise for next iteration (only in self mode)
        if rcfg_type == "self":
            state["stock_noise"] = pos_cond.detach()  # Store current positive cond
            
        state["timestep_count"] += 1
        self.set_state(state)
        
        # Return lists of tuples in ComfyUI's format
        return (
            [(pos_cond, pos_meta)],  # Positive conditioning
            [(residual_neg_cond, neg_meta)]  # Negative conditioning with residual
        )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always execute if enabled to maintain state between steps
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
                "batched_latents": ("LATENT",),  # Buffered latents from StreamDiffusionLatentBuffer
                "t_index_list": ("STRING", {
                    "default": "0,16,32,45",
                    "tooltip": "Comma-separated list of timestep indices to use"
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale"
                }),
                "scheduler": (["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "update"
    CATEGORY = "real-time/sampling"
    DESCRIPTION = "StreamDiffusion's optimized sampler for real-time generation"

    def update(self, model, positive, negative, batched_latents, t_index_list, cfg_scale, scheduler, seed, always_execute=True):
        device = comfy.model_management.get_torch_device()
        buffered_latents = batched_latents["samples"].to(device)  # Shape: [buffer_size, 4, H, W]
        
        # Parse t_index_list from string
        t_indices = [int(x.strip()) for x in t_index_list.split(",")]
        
        # Create sampler object
        sampler = comfy.samplers.sampler_object(scheduler)
        
        # Calculate sigmas for timesteps
        timesteps = torch.linspace(999, 0, max(t_indices) + 2, device=device)[:-1]  # +2 for safety
        sigmas = []
        for t_idx in t_indices:
            sigma = model.model.model_sampling.sigma(torch.tensor([timesteps[t_idx]], device=device))
            sigmas.append(sigma[0])
        sigmas = torch.stack(sigmas)
        
        # Sample using ComfyUI's infrastructure
        torch.manual_seed(seed)
        noise = torch.randn_like(buffered_latents, device=device)
        out = comfy.samplers.sample(model, noise, positive, negative, cfg_scale,
                                  device, sampler, sigmas, latent_image=buffered_latents)
        
        # Return denoised latents - take only the first frame
        return ({"samples": out[:1]},)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return super().IS_CHANGED(**kwargs)