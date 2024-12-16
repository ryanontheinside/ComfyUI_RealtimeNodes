import numpy as np
import torch

class QuickShapeMask:
    """A node that quickly generates basic shape masks"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape": (["circle", "square"], {
                    "default": "circle",
                    "tooltip": "The shape of the mask to generate"
                }),
                "width": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Width of the shape in pixels"
                }),
                "height": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Height of the shape in pixels"
                }),
                "x": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "X position of the shape center (0 = left edge)"
                }),
                "y": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Y position of the shape center (0 = top edge)"
                }),
                "canvas_width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Width of the output mask"
                }),
                "canvas_height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Height of the output mask"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Number of identical masks to generate"
                })
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "generate_mask"
    CATEGORY = "mask"
    
    DESCRIPTION = "Generates a mask containing a basic shape (circle or square) with high performance"

    def generate_mask(self, shape, width, height, x, y, canvas_width, canvas_height, batch_size):
        # Create empty mask
        mask = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        
        # Calculate boundaries
        half_width = width // 2
        half_height = height // 2
        
        # Calculate shape boundaries
        left = max(0, x - half_width)
        right = min(canvas_width, x + half_width)
        top = max(0, y - half_height)
        bottom = min(canvas_height, y + half_height)
        
        if shape == "square":
            # Simple square mask
            mask[top:bottom, left:right] = 1.0
            
        else:  # circle
            # Create coordinate grids for the region of interest
            Y, X = np.ogrid[top:bottom, left:right]
            
            # Calculate distances from center for the region
            dist_x = (X - x)
            dist_y = (Y - y)
            
            # Create circle mask using distance formula
            circle_mask = (dist_x**2 / (width/2)**2 + dist_y**2 / (height/2)**2) <= 1
            
            # Apply circle to the region
            mask[top:bottom, left:right][circle_mask] = 1.0
        
        # Convert to torch tensor and add batch dimension (BHW format)
        mask_tensor = torch.from_numpy(mask)
        
        # Expand to requested batch size
        mask_tensor = mask_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        
        return (mask_tensor,)