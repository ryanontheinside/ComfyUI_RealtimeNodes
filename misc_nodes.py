import torch
import comfy.utils

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
