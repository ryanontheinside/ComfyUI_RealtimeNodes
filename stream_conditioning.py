import torch
from .base.control_base import ControlNodeBase
import comfy.model_management
import comfy.samplers
import random
import math

#NOTE: totally and utterly experimental. No theoretical backing whatsoever.
class StreamConditioning(ControlNodeBase):
    """Applies Residual CFG to conditioning for improved temporal consistency with different CFG types"""
    #NOTE: experimental
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg_type": (["full", "self", "initialize"], {
                    "default": "full",
                    "tooltip": "Type of CFG to use: full (standard), self (memory efficient), or initialize (memory efficient with initialization)"
                }),
                "residual_scale": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Scale factor for residual conditioning (higher = more temporal consistency)"
                }),
                "delta": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Delta parameter for self/initialize CFG types"
                }),
                "context_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Number of past conditionings to keep in context. Higher values = smoother transitions but more memory usage."
                }),
                "always_execute": ("BOOLEAN", {
                    "default": False,
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "update"
    CATEGORY = "real-time/control/utility"

    def __init__(self):
        super().__init__()

    def update(self, positive, negative, cfg_type="full", residual_scale=0.4, delta=1.0, context_size=4, always_execute=False):
        # Get state with defaults
        state = self.get_state({
            "prev_positive": None,
            "prev_negative": None,
            "stock_noise": None,  # For self/initialize CFG
            "initialized": False,  # For initialize CFG
            "pos_context": [],    # Store past positive conditionings
            "neg_context": []     # Store past negative conditionings
        })

        # Extract conditioning tensors
        current_pos_cond = positive[0][0]
        current_neg_cond = negative[0][0]
        
        # Update context queues
        if len(state["pos_context"]) >= context_size:
            state["pos_context"].pop(0)
            state["neg_context"].pop(0)
        state["pos_context"].append(current_pos_cond.detach().clone())
        state["neg_context"].append(current_neg_cond.detach().clone())
        
        # First frame case
        if state["prev_positive"] is None:
            state["prev_positive"] = current_pos_cond.detach().clone()
            state["prev_negative"] = current_neg_cond.detach().clone()
            if cfg_type == "initialize":
                state["stock_noise"] = current_neg_cond.detach().clone()
            elif cfg_type == "self":
                state["stock_noise"] = current_neg_cond.detach().clone() * delta
            self.set_state(state)
            return (positive, negative)

        # Handle different CFG types
        if cfg_type == "full":
            # Use entire context for smoother transitions
            pos_context = torch.stack(state["pos_context"], dim=0)
            neg_context = torch.stack(state["neg_context"], dim=0)
            
            # Calculate weighted residuals across context
            weights = torch.linspace(0.5, 1.0, len(state["pos_context"]), device=current_pos_cond.device)
            pos_residual = (current_pos_cond - pos_context) * weights.view(-1, 1, 1)
            neg_residual = (current_neg_cond - neg_context) * weights.view(-1, 1, 1)
            
            # Average residuals
            pos_residual = pos_residual.mean(dim=0)
            neg_residual = neg_residual.mean(dim=0)
            
            blended_pos = current_pos_cond + residual_scale * pos_residual
            blended_neg = current_neg_cond + residual_scale * neg_residual
            
            # Update state
            state["prev_positive"] = current_pos_cond.detach().clone()
            state["prev_negative"] = current_neg_cond.detach().clone()
            
            # Reconstruct conditioning format
            positive_out = [[blended_pos, positive[0][1]]]
            negative_out = [[blended_neg, negative[0][1]]]
            
        else:  # self or initialize
            # Calculate residual for positive conditioning
            pos_residual = current_pos_cond - state["prev_positive"]
            blended_pos = current_pos_cond + residual_scale * pos_residual
            
            # Update stock noise based on current prediction
            if cfg_type == "initialize" and not state["initialized"]:
                # First prediction for initialize type
                state["stock_noise"] = current_neg_cond.detach().clone()
                state["initialized"] = True
            else:
                # Update stock noise with temporal consistency
                stock_residual = current_neg_cond - state["stock_noise"]
                state["stock_noise"] = current_neg_cond + residual_scale * stock_residual
            
            # Scale stock noise by delta
            scaled_stock = state["stock_noise"] * delta
            
            # Update state
            state["prev_positive"] = current_pos_cond.detach().clone()
            state["prev_negative"] = scaled_stock.detach().clone()
            
            # Reconstruct conditioning format
            positive_out = [[blended_pos, positive[0][1]]]
            negative_out = [[scaled_stock, negative[0][1]]]

        self.set_state(state)
        return (positive_out, negative_out)