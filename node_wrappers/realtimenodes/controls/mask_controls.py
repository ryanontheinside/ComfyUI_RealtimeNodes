import torch
import numpy as np
import cv2
from ....src.realtimenodes.control_base import ControlNodeBase
from ....src.realtimenodes.mask_controls import MaskControlMixin

class RepulsiveMaskNode(ControlNodeBase, MaskControlMixin):
    """Node that maintains a mask that repulses or attracts to input masks"""
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "input_mask": ("MASK",),
            "x_pos": ("FLOAT", {
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Initial X position (0-1)"
            }),
            "y_pos": ("FLOAT", {
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Initial Y position (0-1)"
            }),
            "size": ("FLOAT", {
                "default": 0.2,
                "min": 0.01,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Size of the mask relative to image size"
            }),
            "interaction_type": (["repulse", "attract"], {
                "default": "repulse",
                "tooltip": "Whether to move away from or toward input masks"
            }),
            "interaction_strength": ("FLOAT", {
                "default": 0.1,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Strength of the interaction effect"
            })
        })
        return inputs

    RETURN_TYPES = ("MASK",)
    FUNCTION = "update"
    CATEGORY = "real-time/control/mask"

    def interact(self, mask, input_mask_np, state, strength, height, width, is_attract):
        if np.max(input_mask_np) < 0.1:
            return state
            
        y_indices, x_indices = np.nonzero(input_mask_np)
        if len(y_indices) == 0:
            return state
            
        weights = input_mask_np[y_indices, x_indices]
        com_y = np.average(y_indices, weights=weights)
        com_x = np.average(x_indices, weights=weights)
        
        center_y = state["y"] * height
        center_x = state["x"] * width
        
        dir_y = center_y - com_y
        dir_x = center_x - com_x
        
        dist = np.sqrt(dir_x**2 + dir_y**2)
        if dist < 1:
            return state
            
        scale = min(1.0, 20.0/dist)
        dir_y = (dir_y / dist) * scale
        dir_x = (dir_x / dist) * scale
        
        # Flip direction if attracting
        if is_attract:
            dir_y *= -1
            dir_x *= -1
        
        new_y = state["y"] + (dir_y * strength)
        new_x = state["x"] + (dir_x * strength)
        
        state["y"] = np.clip(new_y, 0.0, 1.0)
        state["x"] = np.clip(new_x, 0.0, 1.0)
        
        return state

    def update(self, input_mask, x_pos, y_pos, size, interaction_type, interaction_strength, always_execute=True):
        was_reset = len(self.state_manager._states) == 0
        
        if was_reset:
            state = self.get_initial_state(x_pos, y_pos, size)
        else:
            state = self.get_state(self.get_initial_state(x_pos, y_pos, size))

        batch_size, height, width = input_mask.shape
        masks = np.zeros((batch_size, height, width), dtype=np.float32)
            
        for b in range(batch_size):
            center_y, center_x = int(state["y"] * height), int(state["x"] * width)
            mask = self.create_circle_mask(height, width, center_y, center_x, state["size"])
        
            state = self.interact(mask, input_mask[b].cpu().numpy(), state, 
                                interaction_strength, height, width, 
                                interaction_type == "attract")
        
            center_y, center_x = int(state["y"] * height), int(state["x"] * width)
            masks[b] = self.create_circle_mask(height, width, center_y, center_x, state["size"])
        
        self.set_state(state)
        return (torch.from_numpy(masks),)

class ResizeMaskNode(ControlNodeBase, MaskControlMixin):
    """Node that maintains a mask that resizes based on proximity to input masks"""
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "input_mask": ("MASK",),
            "x_pos": ("FLOAT", {
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Initial X position (0-1)"
            }),
            "y_pos": ("FLOAT", {
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Initial Y position (0-1)"
            }),
            "size": ("FLOAT", {
                "default": 0.2,
                "min": 0.01,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Size of the mask relative to image size"
            }),
            "min_size": ("FLOAT", {
                "default": 0.1,
                "min": 0.01,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Minimum size the mask can shrink to"
            }),
            "max_size": ("FLOAT", {
                "default": 0.4,
                "min": 0.01,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Maximum size the mask can grow to"
            }),
            "resize_strength": ("FLOAT", {
                "default": 0.1,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "Strength of the resize effect"
            })
        })
        return inputs

    RETURN_TYPES = ("MASK",)
    FUNCTION = "update"
    CATEGORY = "real-time/control/mask"

    def resize(self, mask, input_mask_np, state, strength, height, width):
        if np.max(input_mask_np) < 0.1:
            # No mask detected - use max size
            state["size"] = state["max_size"]
            return state

        # Get distance from center to nearest mask pixel
        center_y = int(state["y"] * height)
        center_x = int(state["x"] * width)
        y_indices, x_indices = np.nonzero(input_mask_np > 0.1)
        
        if len(y_indices) == 0:
            return state
            
        distances = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
        min_dist = np.min(distances)
        
        # Map distance directly to size
        # When dist=0, size=min_size
        # When dist=100, size=max_size
        dist_range = 100.0  # pixels
        size_range = state["max_size"] - state["min_size"]
        size = state["min_size"] + (min_dist / dist_range) * size_range
        state["size"] = np.clip(size, state["min_size"], state["max_size"])
            
        return state

    def update(self, input_mask, x_pos, y_pos, size, min_size, max_size, resize_strength, always_execute=True):
        was_reset = len(self.state_manager._states) == 0
        
        if was_reset:
            state = self.get_initial_state(x_pos, y_pos, size, min_size, max_size)
        else:
            state = self.get_state(self.get_initial_state(x_pos, y_pos, size, min_size, max_size))

        batch_size, height, width = input_mask.shape
        masks = np.zeros((batch_size, height, width), dtype=np.float32)
            
        for b in range(batch_size):
            center_y, center_x = int(state["y"] * height), int(state["x"] * width)
            mask = self.create_circle_mask(height, width, center_y, center_x, state["size"])
        
            state = self.resize(mask, input_mask[b].cpu().numpy(), state, resize_strength, height, width)
        
            center_y, center_x = int(state["y"] * height), int(state["x"] * width)
            masks[b] = self.create_circle_mask(height, width, center_y, center_x, state["size"])
        
        self.set_state(state)
        return (torch.from_numpy(masks),) 