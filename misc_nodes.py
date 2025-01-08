import torch
import comfy.utils
import numpy as np
import base64
import re
from io import BytesIO
from PIL import Image
import nodes
import random
from torchvision import transforms

MAX_RESOLUTION = nodes.MAX_RESOLUTION  # Get the same max resolution as core nodes

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

class DTypeConverter:
    """Converts masks to specified data types"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),  # Explicitly accept only MASK input
                "dtype": (["float16", "uint8", "float32", "float64"],),
            }
        }

    CATEGORY = "utils"
    RETURN_TYPES = ("MASK",)  # Return only MASK type
    FUNCTION = "convert_dtype"

    DTYPE_MAP = {
        "uint8": torch.uint8,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64
    }

    def convert_dtype(self, mask, dtype):
        target_dtype = self.DTYPE_MAP[dtype]
        
        if target_dtype == torch.uint8:
            if mask.is_floating_point():
                converted = (mask * 255.0).round().to(torch.uint8)
            else:
                converted = (mask > 0).to(torch.uint8) * 255
        else:  # Converting to float
            if mask.dtype == torch.uint8:
                converted = (mask > 0).to(target_dtype)
            else:
                converted = mask.to(target_dtype)
        
        return (converted,)   


class FastWebcamCapture:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("WEBCAM", {}),
                "width": ("INT", {"default": 640, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 480, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "capture_on_queue": ("BOOLEAN", {"default": True}),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_capture"
    
    CATEGORY = "image"

    def process_capture(self, image, width, height, capture_on_queue):
        # Check if we got a data URL
        if isinstance(image, str) and image.startswith('data:image/'):
            # Extract the base64 data after the comma
            base64_data = re.sub('^data:image/.+;base64,', '', image)
            
            # Convert base64 to PIL Image
            buffer = BytesIO(base64.b64decode(base64_data))
            pil_image = Image.open(buffer).convert("RGB")
            
            # Convert PIL to numpy array
            image = np.array(pil_image)
            
            # Handle resize if requested
            if width > 0 and height > 0:
                import cv2
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            
            # Convert to float32 and normalize to 0-1 range
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension and convert to torch tensor
            # ComfyUI expects BHWC format
            image = torch.from_numpy(image)[None,...]
            
            return (image,)
        else:
            raise ValueError("Invalid image format received from webcam")

