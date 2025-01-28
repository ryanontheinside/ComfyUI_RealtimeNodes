from ..base.control_base import ControlNodeBase
import time
import numpy as np
import cv2
import torch
import random

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

class SimilarityFilter(ControlNodeBase):
    """A node that filters out similar consecutive images to prevent unnecessary workflow execution using StreamDiffusion's approach."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "always_execute": ("BOOLEAN", {"default": False}),
                "threshold": ("FLOAT", {"default": 0.98, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_skip_frames": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "update"
    CATEGORY = "real-time/control/utility"

    def __init__(self):
        super().__init__()
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    def update(self, image, threshold=0.98, max_skip_frames=10, always_execute=False):
        print(f"[DEBUG] Input image object: {hex(id(image))}, shape: {image.shape}, device: {image.device}")
        
        # Get state with defaults
        state = self.get_state({
            "prev_image": None,
            "skip_count": 0
        })

        # First frame case
        if state["prev_image"] is None:
            state["prev_image"] = image.detach().clone()
            state["skip_count"] = 0
            self.set_state(state)
            return (image,)

        # Calculate cosine similarity
        cos_sim = self.cos(
            state["prev_image"].reshape(-1), 
            image.reshape(-1)
        ).item()

        # Calculate skip probability using StreamDiffusion's approach
        if threshold >= 1:
            skip_prob = 0
        else:
            skip_prob = max(0, 1 - (1 - cos_sim) / (1 - threshold))

        # Generate random sample
        sample = random.uniform(0, 1)

        # If we should skip (probability check)
        if skip_prob >= sample:
            # Check if we've skipped too many frames
            if state["skip_count"] >= max_skip_frames:
                state["prev_image"] = image.detach().clone()
                state["skip_count"] = 0
                self.set_state(state)
                return (image,)
                
            # Skip frame - return ExecutionBlocker to prevent downstream execution
            state["skip_count"] += 1
            self.set_state(state)
            from comfy_execution.graph import ExecutionBlocker
            return (ExecutionBlocker(None),)
        
        # Frame is different enough - process it
        state["prev_image"] = image.detach().clone()
        state["skip_count"] = 0
        self.set_state(state)
        return (image,)
