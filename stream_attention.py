import torch
import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention

class StreamCrossAttention:
    original_attention = None
    
    @classmethod
    def get_attention_scores(cls, q, k, v, extra_options):
        """
        Optimized attention computation with:
        - FlashAttention when available
        - KV cache preservation
        - Sliced attention fallback
        """
        # Get transformer options
        transformer_options = extra_options.get("transformer_options", {})
        
        # Get cached key/values if available
        kv_cache = transformer_options.get("kv_cache", {})
        layer_id = extra_options["layer_id"]
        
        # Check if we should use cached KV
        if kv_cache is not None and layer_id in kv_cache:
            cached = kv_cache[layer_id]
            k = cached["k"]
            v = cached["v"]
        else:
            # Store KV in cache if enabled
            if kv_cache is not None:
                kv_cache[layer_id] = {"k": k, "v": v}

        # Use ComfyUI's memory-efficient attention
        return optimized_attention(
            q, k, v, 
            heads=extra_options["n_heads"],
            mask=extra_options.get("mask"),
        )

    @classmethod
    def initialize_attention(cls):
        """Patch ComfyUI's attention modules with our optimized version"""
        from comfy.ldm.modules.attention import CrossAttention
        
        if cls.original_attention is None:
            cls.original_attention = CrossAttention.forward
        
        def stream_attention_forward(self, x, context=None, mask=None, **kwargs):
            q = self.to_q(x)
            context = context if context is not None else x
            k = self.to_k(context)
            v = self.to_v(context)
            
            out = cls.get_attention_scores(
                q, k, v,
                extra_options={
                    "n_heads": self.heads,
                    "layer_id": id(self),
                    "mask": mask,
                    "transformer_options": kwargs.get("transformer_options", {})
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
        self.current_batch_size = None
        
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
        
        # Get current batch size
        batch_size = x.shape[0]
        
        # Reinitialize noise base if batch size changes
        if self.current_batch_size != batch_size:
            self.noise_base = None
            self.current_batch_size = batch_size
            
        # Generate shared noise pattern for batch
        if self.noise_base is None:
            self.noise_base = torch.randn_like(x)
        
        # Get transformer options
        transformer_options = c.get("transformer_options", {})
        
        # Add KV cache to transformer options
        transformer_options["kv_cache"] = self.kv_cache
        
        # Pass through model with updated options
        return apply_model(
            x,
            timestep,
            c_concat=c.get("c_concat"),
            c_crossattn=c.get("c_crossattn"),
            control=c.get("control"),
            transformer_options=transformer_options,
            cond_or_uncond=args.get("cond_or_uncond", None)
        )
