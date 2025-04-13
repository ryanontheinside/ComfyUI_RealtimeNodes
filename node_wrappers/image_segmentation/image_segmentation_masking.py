import torch
import numpy as np
import logging


logger = logging.getLogger(__name__)
_category = "MediaPipeVision/ImageSegmentation/Masking"
# Define class names and their corresponding indices for multiclass models
MULTICLASS_NAMES = {
    "Background": 0,
    "Hair": 1,
    "Body-skin": 2,
    "Face-skin": 3,
    "Clothes": 4,
    "Accessories/Other": 5,
}

class SelectMediaPipeSegmentNode:
    """
    Selects a specific segmentation mask from the raw multiclass output
    of the MediaPipeImageSegmenterNode.
    """
    @classmethod
    def INPUT_TYPES(cls):
        available_segments = list(MULTICLASS_NAMES.keys())
        return {
            "required": {
                "multiclass_segments": ("MP_INT_MASK", { 
                    "tooltip": "Raw multiclass segmentation data (INT_MASK) from MediaPipe Image Segmenter node."
                }),
                "segment_name": (available_segments, { 
                    "default": "Clothes" if "Clothes" in available_segments else available_segments[0],
                    "tooltip": "Select the specific segment to extract as a mask."
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "select_segment"
    CATEGORY = _category
    DESCRIPTION = "Extracts a single mask for a selected segment (e.g., Hair, Clothes) from the multiclass segmentation data."

    def select_segment(self, multiclass_segments: torch.Tensor, segment_name: str):
        if segment_name not in MULTICLASS_NAMES:
            raise ValueError(f"Invalid segment_name '{segment_name}'. Available: {list(MULTICLASS_NAMES.keys())}")
        
        if multiclass_segments is None or multiclass_segments.numel() == 0:
             logger.warning("Input multiclass_segments is empty or None. Returning zero mask.")
             # Need a shape to return a zero mask. How to get expected shape?
             # For now, maybe raise error or return None (caller must handle)
             # Returning None is safer if shape is unknown.
             return (None,) # Or raise error
        
        # Check tensor format and reshape if needed
        # Input formats can be either BHW or B1HW (both are common in ComfyUI)
        if multiclass_segments.dim() == 3:  # BHW format
            # Reshape to B1HW format
            logger.info(f"Reshaping multiclass_segments from BHW format {multiclass_segments.shape} to B1HW format")
            multiclass_segments = multiclass_segments.unsqueeze(1)  # Insert channel dimension
            logger.info(f"New shape: {multiclass_segments.shape}")
        elif multiclass_segments.dim() == 4 and multiclass_segments.shape[1] != 1:
            # If it's BCHW but not B1HW, raise error
            raise ValueError(f"Expected multiclass_segments to be B1HW or BHW tensor. Got shape {multiclass_segments.shape}")
        
        # Ensure input is a LongTensor
        if multiclass_segments.dtype != torch.int64:
            multiclass_segments = multiclass_segments.to(torch.int64)

        target_id = MULTICLASS_NAMES[segment_name]
        selected_mask = (multiclass_segments == target_id).float()
        
        # If result is B1HW, squeeze back to BHW for ComfyUI mask format compatibility
        if selected_mask.dim() == 4:
            selected_mask = selected_mask.squeeze(1)
            
        return (selected_mask,)
# --- End Select Segment Node --- 

# --- Mappings --- 
NODE_CLASS_MAPPINGS = {
    "SelectMediaPipeSegment": SelectMediaPipeSegmentNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelectMediaPipeSegment": "Select MediaPipe Segment",
}
