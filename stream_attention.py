import torch
import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention

class StreamCrossAttention:
    @classmethod
    def get_attention_scores(cls, q, k, v, extra_options):
        """
        Optimized attention computation with:
        - FlashAttention when available
        - KV cache preservation
        - Sliced attention fallback
        """
        print(f"[DEBUG] Attention input q stats - min: {q.min():.4f}, max: {q.max():.4f}, mean: {q.mean():.4f}")
        print(f"[DEBUG] Attention input k stats - min: {k.min():.4f}, max: {k.max():.4f}, mean: {k.mean():.4f}")
        print(f"[DEBUG] Attention input v stats - min: {v.min():.4f}, max: {v.max():.4f}, mean: {v.mean():.4f}")
        
        if not hasattr(cls, "attention_impl"):
            # Initialize optimized implementation once
            cls.attention_impl = cls.initialize_attention()
        
        # Get cached key/values if available
        cache = extra_options.get("_stream_kv_cache", {})
        layer_id = extra_options["layer_id"]
        
        if layer_id not in cache:
            cache[layer_id] = {"k": None, "v": None}
        
        # Use cached KV if available
        if cache[layer_id]["k"] is not None:
            k = cache[layer_id]["k"]
            v = cache[layer_id]["v"]
            print(f"[DEBUG] Using cached k stats - min: {k.min():.4f}, max: {k.max():.4f}, mean: {k.mean():.4f}")
            print(f"[DEBUG] Using cached v stats - min: {v.min():.4f}, max: {v.max():.4f}, mean: {v.mean():.4f}")
        else:  # First run or cache disabled
            cache[layer_id]["k"] = k
            cache[layer_id]["v"] = v

        # Use ComfyUI's memory-efficient attention as base
        output = optimized_attention(
            q, k, v, 
            heads=extra_options["n_heads"],
            mask=extra_options["mask"],
            attn_precision=None,  # Let ComfyUI handle precision
            skip_reshape=False,  # Default reshape behavior
            skip_output_reshape=False  # Default output reshape behavior
        )
        print(f"[DEBUG] Attention output stats - min: {output.min():.4f}, max: {output.max():.4f}, mean: {output.mean():.4f}")
        return output

    @classmethod
    def initialize_attention(cls):
        """Patch ComfyUI's attention modules with our optimized version"""
        from comfy.ldm.modules.attention import CrossAttention
        original_attention = CrossAttention.forward
        
        def stream_attention_forward(self, x, context=None, mask=None, **kwargs):
            # Convert to BHWC format if needed
            if x.shape[1] == self.dim_head * self.heads:
                x = x.view(x.shape[0], -1, self.dim_head).permute(0, 2, 1, 3)
            
            # Use our optimized attention implementation
            q = self.to_q(x)
            context = context if context is not None else x
            k = self.to_k(context)
            v = self.to_v(context)
            
            out = cls.get_attention_scores(
                q, k, v,
                extra_options={
                    "n_heads": self.heads,
                    "layer_id": id(self),  # Unique ID for KV caching
                    "mask": mask
                }
            )
            return self.to_out(out)
        
        # Monkey-patch ComfyUI's attention
        CrossAttention.forward = stream_attention_forward
        return stream_attention_forward

class UNetBatchedStream:
    """
    Wrapper for ComfyUI's model to enable:
    - Batched denoising across multiple frames
    - Attention KV caching between denoising steps
    - Shared noise base pattern
    """
    def __init__(self, model):
        self.inner_model = model
        self.kv_cache = {}
        self.noise_base = None
        self.current_channels = None  # Track channel count
        
    def __call__(self, apply_model, args):
        """
        Proper ComfyUI interface implementation
        args = {
            "input": x,
            "timestep": timestep,
            "c": c,
            "cond_or_uncond": cond_or_uncond
        }
        """
        x = args["input"]
        timestep = args["timestep"]
        c = args["c"]
        
        print(f"[DEBUG] Initial x stats - min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}")
        
        # Always convert to BHWC regardless of channel count
        if x.ndim == 4 and x.shape[1] != x.shape[-1]:  # BCHW format
            x = x.permute(0, 2, 3, 1)
            print(f"[DEBUG] After permute - min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}")
            
        # Get current channel count from LAST dimension
        current_channels = x.shape[-1]
        
        # Reinitialize noise base if channels change
        if self.current_channels != current_channels:
            self.noise_base = None
            self.current_channels = current_channels
            
        # Generate shared noise pattern matching current channels
        if self.noise_base is None:
            self.noise_base = torch.randn_like(x)
            print(f"[DEBUG] Generated noise_base - min: {self.noise_base.min():.4f}, max: {self.noise_base.max():.4f}, mean: {self.noise_base.mean():.4f}")
            
        # Apply noise blending
        x = self.blend_noise(x, timestep)
        print(f"[DEBUG] After blend_noise - min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}")
        
        # Convert back to BCHW for ComfyUI
        x = x.permute(0, 3, 1, 2)
        print(f"[DEBUG] Final x stats - min: {x.min():.4f}, max: {x.max():.4f}, mean: {x.mean():.4f}")
        
        # Extract tensors from conditioning
        c_concat = c.get('c_concat', None)
        if isinstance(c_concat, dict) and 'cond' in c_concat:
            c_concat = c_concat['cond']
            
        c_crossattn = c.get('c_crossattn', None)
        if isinstance(c_crossattn, dict) and 'cond' in c_crossattn:
            c_crossattn = c_crossattn['cond']
            
        # Run through original apply_model with modified args
        output = apply_model(
            x,
            timestep,
            c_concat,
            c_crossattn,
            cond_or_uncond=args.get("cond_or_uncond", None),
            kv_cache=self.kv_cache
        )
        print(f"[DEBUG] Output stats - min: {output.min():.4f}, max: {output.max():.4f}, mean: {output.mean():.4f}")
        return output
    
    def blend_noise(self, x, timesteps):
        """Handle different channel counts in noise blending"""
        print(f"[DEBUG] x shape: {x.shape}")
        print(f"[DEBUG] noise_base shape: {self.noise_base.shape if self.noise_base is not None else None}")
        print(f"[DEBUG] x device: {x.device}")
        print(f"[DEBUG] noise_base device: {self.noise_base.device if self.noise_base is not None else None}")
        
        # Ensure noise_base matches x's channels
        if self.noise_base is None or self.noise_base.shape[-1] != x.shape[-1]:
            print(f"[DEBUG] Regenerating noise_base to match shape")
            self.noise_base = torch.randn_like(x)
            print(f"[DEBUG] New noise_base shape: {self.noise_base.shape}")
            
        alpha = 1.0 / (1.0 + (timesteps / 0.1) ** 2)
        print(f"[DEBUG] alpha value: {alpha}, shape: {alpha.shape if hasattr(alpha, 'shape') else 'scalar'}")
        
        # Ensure alpha broadcasts correctly
        if torch.is_tensor(alpha):
            alpha = alpha.view(-1, 1, 1, 1)
        else:
            alpha = torch.tensor(alpha, device=x.device).view(-1, 1, 1, 1)
            
        print(f"[DEBUG] reshaped alpha shape: {alpha.shape}")
        return alpha * x + (1 - alpha) * self.noise_base