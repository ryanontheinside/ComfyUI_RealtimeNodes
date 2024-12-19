from ..base.control_base import ControlNodeBase
import time
import numpy as np
import cv2
import torch

class FPSMonitor(ControlNodeBase):
    """Generates an FPS overlay as an image and mask"""
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "update"
    CATEGORY = "real-time/control/utility"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "width": ("INT", {
                "default": 512,
                "min": 64,
                "max": 4096,
                "step": 1
            }),
            "height": ("INT", {
                "default": 512,
                "min": 64,
                "max": 4096,
                "step": 1
            }),
            "text_color": ("INT", {
                "default": 255,
                "min": 0,
                "max": 255,
                "step": 1
            }),
            "text_size": ("FLOAT", {
                "default": 1.0,
                "min": 0.1,
                "max": 5.0,
                "step": 0.1
            }),
            "window_size": ("INT", {
                "default": 60,
                "min": 1,
                "max": 300,
                "step": 1,
                "tooltip": "Number of frames to average over"
            })
        })
        return inputs

    def update(self, width: int, height: int, text_color: int, text_size: float, window_size: int, always_execute=True):
        current_time = time.time()
        state = self.get_state({
            "last_time": current_time,
            "frame_times": [],
            "last_fps": 0,
            "last_avg_fps": 0,
            "cached_image": None,
            "cached_mask": None,
            "last_dims": (0, 0),
            "last_color": 0,
            "last_size": 0
        })
        
        # Calculate FPS
        delta_time = current_time - state["last_time"]
        current_fps = 1.0 / delta_time if delta_time > 0 else 0
        
        # Update rolling window
        state["frame_times"].append(delta_time)
        if len(state["frame_times"]) > window_size:
            state["frame_times"].pop(0)
            
        # Calculate average FPS
        avg_delta = sum(state["frame_times"]) / len(state["frame_times"])
        average_fps = 1.0 / avg_delta if avg_delta > 0 else 0
        
        # Check if we need to redraw
        needs_redraw = (
            abs(current_fps - state["last_fps"]) > 1.0 or  # FPS changed significantly
            abs(average_fps - state["last_avg_fps"]) > 0.5 or  # Avg FPS changed significantly
            state["last_dims"] != (width, height) or  # Dimensions changed
            state["last_color"] != text_color or  # Color changed
            state["last_size"] != text_size or  # Text size changed
            state["cached_image"] is None  # No cache exists
        )
        
        if needs_redraw:
            # Create text
            fps_text = f"FPS: {current_fps:.1f} (avg: {average_fps:.1f})"
            
            # Create image and mask using numpy/cv2
            image = np.zeros((height, width, 3), dtype=np.uint8)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Position text in top-left with padding
            padding = 10
            pos = (padding, padding + 20)  # Add vertical offset for cv2's text positioning
            
            # Draw text - much simpler with cv2!
            cv2.putText(image, fps_text, pos, cv2.FONT_HERSHEY_SIMPLEX, text_size, (text_color, text_color, text_color), 1)
            cv2.putText(mask, fps_text, pos, cv2.FONT_HERSHEY_SIMPLEX, text_size, 255, 1)
            
            # Convert to float32 and normalize
            image = image.astype(np.float32) / 255.0
            mask = mask.astype(np.float32) / 255.0
            
            # Convert to torch tensors with correct shapes
            state["cached_image"] = torch.from_numpy(image[None, ...])  # BHWC
            state["cached_mask"] = torch.from_numpy(mask[None, ...])    # BHW
            
            # Update cache parameters
            state["last_fps"] = current_fps
            state["last_avg_fps"] = average_fps
            state["last_dims"] = (width, height)
            state["last_color"] = text_color
            state["last_size"] = text_size
        
        state["last_time"] = current_time
        self.set_state(state)
        
        return (state["cached_image"], state["cached_mask"])
