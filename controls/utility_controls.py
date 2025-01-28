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
    DESCRIPTION = "Filters out similar consecutive images and outputs a signal to control downstream execution."
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Input image to compare with previous frame"
                }),
                "always_execute": ("BOOLEAN", {
                    "default": False,
                }),
                "threshold": ("FLOAT", {
                    "default": 0.98,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Similarity threshold (0-1). Higher values mean more frames are considered similar"
                }),
                "max_skip_frames": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Maximum number of consecutive frames to skip before forcing execution"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "should_execute")
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
            return (image, True)  # Always execute first frame

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
                return (image, True)  # Force execution after max skips
                
            # Skip frame - return previous image and False for execution
            state["skip_count"] += 1
            self.set_state(state)
            return (state["prev_image"], False)
        
        # Frame is different enough - process it
        state["prev_image"] = image.detach().clone()
        state["skip_count"] = 0
        self.set_state(state)
        return (image, True)

class LazyCondition(ControlNodeBase):
    DESCRIPTION = "Uses lazy evaluation to truly skip execution of unused paths. Maintains state of the last diffused image to avoid feedback loops."
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "condition": ("BOOLEAN", {
                    "tooltip": "When True, evaluates and returns if_true path. When False, returns either fallback or previous state"
                }),
                "if_true": ("IMAGE", {
                    "lazy": True,
                    "tooltip": "The expensive computation path (e.g., diffusion) that should only be evaluated when condition is True"
                }),
                "fallback": ("IMAGE", {
                    "tooltip": "Alternative image to use when condition is False (e.g., input frame)"
                }),
                "use_fallback": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When False, uses last successful if_true result. When True, uses fallback image"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "update"
    CATEGORY = "real-time/control/utility"

    def check_lazy_status(self, condition, if_true, fallback, use_fallback):
        """Only evaluate the diffusion path if condition is True."""
        needed = ["fallback"]  # Always need the input image
        if condition:
            needed.append("if_true")  # Only compute diffusion if needed
        return needed

    def update(self, condition, if_true, fallback, use_fallback):
        """Route to either diffusion output or input image, maintaining state of last diffusion."""
        state = self.get_state({
            "prev_output": None
        })

        if condition:
            print("[LazyCondition] Condition True - Running diffusion path")
            # Update last diffused state when we run diffusion
            state["prev_output"] = if_true.detach().clone() if if_true is not None else fallback
            self.set_state(state)
            return (if_true,)
        else:
            if use_fallback or state["prev_output"] is None:
                print("[LazyCondition] Condition False - Using fallback image")
                return (fallback,)
            else:
                print("[LazyCondition] Condition False - Using previous diffusion result")
                return (state["prev_output"],)
