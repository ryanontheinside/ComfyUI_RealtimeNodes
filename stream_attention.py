import torch
from .base.control_base import ControlNodeBase
import comfy.model_management
import comfy.samplers
import random
import math

class StreamCrossAttention(ControlNodeBase):
    DESCRIPTION="""Implements optimized cross attention with KV-cache for real-time generation
    
    Paper reference: StreamDiffusion Section 3.5 "Pre-computation"
    - Pre-computes and caches prompt embeddings
    - Stores Key-Value pairs for reuse with static prompts
    - Only recomputes KV pairs when prompt changes
    """
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "update"
    CATEGORY = "real-time/sampling"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        del inputs["required"]["always_execute"]
        inputs["required"].update({
            "model": ("MODEL",),
            "prompt": ("STRING", {
                "multiline": True,
                "forceInput": True,
                "default": "",
                "tooltip": "Text prompt to use for caching. Only recomputes when this changes."
            }),
            "max_cache_size": ("INT", {
                "default": 8,
                "min": 1,
                "max": 32,
                "step": 1,
                "tooltip": "Maximum number of cached entries per module"
            }),
        })
        return inputs

    def __init__(self):
        super().__init__()
        self.cross_attention_hook = None

    def update(self, model, prompt="", max_cache_size=8):
        print(f"[StreamCrossAttention] Initializing with prompt='{prompt}', max_cache_size={max_cache_size}")
        
        # NOTE: Unlike the StreamDiffusion paper, we don't explicitly compare prompts to detect changes.
        # Instead, we leverage ComfyUI's execution system:
        # 1. Our node only executes when inputs (model, prompt, max_cache_size) change
        # 2. When executed, we get the new prompt value and automatically recompute KV pairs
        # 3. The cache key system using (module_id, prompt) ensures we use the right KVs

        # This is more efficient as we avoid explicit prompt comparison and let ComfyUI handle change detection.
        # We do not expect the model or max cache size to change often.
        
        state = self.get_state({
            "max_cache_size": max_cache_size,
            "workflow_count": 0,
            "kv_cache": {},  # From paper Section 3.5: Cache KV pairs for each prompt
            "cache_keys_by_module": {},  # Track cache keys per module for LRU eviction
        })
        
        def manage_cache_size(module_id):
            """Maintain cache size limits using LRU eviction"""
            if module_id in state["cache_keys_by_module"]:
                module_keys = state["cache_keys_by_module"][module_id]
                while len(module_keys) > state["max_cache_size"]:
                    # Remove oldest cache entry
                    old_key = module_keys.pop(0)
                    if old_key in state["kv_cache"]:
                        del state["kv_cache"][old_key]
        
        def get_cache_key(module, prompt):
            """Generate cache key from module ID and prompt text"""
            return (id(module), prompt)
        
        def cross_attention_forward(module, x, context=None, mask=None, value=None):
            """Optimized cross attention following StreamDiffusion's approach"""
            batch_size = x.shape[0]
            context = x if context is None else context
            
            # Debug cache hit/miss
            cache_key = get_cache_key(module, prompt)
            cache_hit = cache_key in state["kv_cache"]
            print(f"[StreamCrossAttn] Cache {'hit' if cache_hit else 'miss'} for module {id(module)}")
            print(f"[StreamCrossAttn] Cache key: {cache_key}")
            
            # Check cache
            if cache_hit:
                k, v = state["kv_cache"][cache_key]
                print(f"[StreamCrossAttn] Reusing cached KV pairs shape k:{k.shape} v:{v.shape}")
                
                # Update LRU tracking
                module_keys = state["cache_keys_by_module"].get(id(module), [])
                if cache_key in module_keys:
                    module_keys.remove(cache_key)
                module_keys.append(cache_key)
                state["cache_keys_by_module"][id(module)] = module_keys
            else:
                # Generate new KV pairs
                k = module.to_k(context)
                v = value if value is not None else module.to_v(context)
                print(f"[StreamCrossAttn] Generated new KV pairs shape k:{k.shape} v:{v.shape}")
                
                # Cache without cloning - just use references
                state["kv_cache"][cache_key] = (k, v)
                
                # Update LRU tracking
                module_keys = state["cache_keys_by_module"].get(id(module), [])
                module_keys.append(cache_key)
                state["cache_keys_by_module"][id(module)] = module_keys
                
                # Basic LRU cleanup
                if len(module_keys) > state["max_cache_size"]:
                    old_key = module_keys.pop(0)
                    if old_key in state["kv_cache"]:
                        print(f"[StreamCrossAttn] Evicting old cache key {old_key}")
                        del state["kv_cache"][old_key]
            
            # Generate query
            q = module.to_q(x)
            print(f"[StreamCrossAttn] Generated query shape:{q.shape}")
            
            # Efficient single-pass reshape
            head_dim = q.shape[-1] // module.heads
            q = q.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)
            k = k.view(-1, k.shape[1], module.heads, head_dim).transpose(1, 2)
            v = v.view(-1, v.shape[1], module.heads, head_dim).transpose(1, 2)
            
            # Handle batch size expansion if needed
            if k.shape[0] == 1 and batch_size > 1:
                print(f"[StreamCrossAttn] Expanding cached KV pairs from batch 1 to {batch_size}")
                k = k.expand(batch_size, -1, -1, -1)
                v = v.expand(batch_size, -1, -1, -1)
            
            # Simple attention computation
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            if mask is not None:
                scores = scores + mask
            
            # Compute attention and output
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            print(f"[StreamCrossAttn] Attention output shape:{out.shape}")
            
            # Final reshape
            out = out.transpose(1, 2).reshape(batch_size, -1, module.heads * head_dim)
            
            return module.to_out[0](out)
        
        def hook_cross_attention(module, input, output):
            if isinstance(module, torch.nn.Module) and hasattr(module, "to_q"):
                # Store original forward
                if not hasattr(module, "_original_forward"):
                    module._original_forward = module.forward
                # Replace with our optimized version
                module.forward = lambda *args, **kwargs: cross_attention_forward(module, *args, **kwargs)
            return output

        # Remove old hooks if they exist
        if self.cross_attention_hook is not None:
            self.cross_attention_hook.remove()
        
        # Clone model and apply hooks
        m = model.clone()
        
        # Register hook for cross attention modules
        def register_hooks(module):
            if isinstance(module, torch.nn.Module) and hasattr(module, "to_q"):
                self.cross_attention_hook = module.register_forward_hook(hook_cross_attention)
        
        m.model.apply(register_hooks)
        return (m,)




