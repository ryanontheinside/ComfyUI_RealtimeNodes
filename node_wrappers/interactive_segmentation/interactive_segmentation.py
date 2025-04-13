import torch
import logging
import json

# Import Base Loader and Detector
from ..common.model_loader import MediaPipeModelLoaderBaseNode
from ...src.interactive_segmentation.detector import InteractiveSegmenterProcessor
from ...src.types import PointOfInterest

logger = logging.getLogger(__name__)
_category = "MediaPipeVision/InteractiveSegmentation"
# --- Model Loader --- 
class MediaPipeInteractiveSegmenterModelLoaderNode(MediaPipeModelLoaderBaseNode):
    """ComfyUI node for loading MediaPipe Interactive Segmenter models."""
    # Usually uses a specific model like selfie_segmenter or a custom one
    TASK_TYPE = "interactive_segmenter" # Need to define this task and models in model_loader.py
    RETURN_TYPES = ("INTERACTIVE_SEGMENTER_MODEL_INFO",)
    RETURN_NAMES = ("model_info",)
    CATEGORY = _category
# --- Segmenter Node --- 
class MediaPipeInteractiveSegmenterNode:
    """ComfyUI node for MediaPipe Interactive Image Segmentation."""

    def __init__(self):
        self._processor: InteractiveSegmenterProcessor = None
        self._model_path: str = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_info": ("INTERACTIVE_SEGMENTER_MODEL_INFO",),
                # Input points as a JSON string list of lists/tuples: e.g., "[[0.5, 0.5], [0.2, 0.3]]"
                "points_json": ("STRING", {"default": "[[0.5, 0.5]]", "multiline": True,
                                            "tooltip": "Points of interest (normalized coords) as JSON: [[x1,y1], [x2,y2], ...]"}),
                "output_category_mask": ("BOOLEAN", {"default": True, 
                                                  "tooltip": "Output a category mask (0=background, 1=foreground). Priority over confidence."}),
                "output_confidence_mask": ("BOOLEAN", {"default": False,
                                                   "tooltip": "Output a confidence mask (0.0-1.0). Used if category mask is False."}),
            }
        }

    RETURN_TYPES = ("MASK",) # Output is a standard ComfyUI mask
    RETURN_NAMES = ("mask",)
    FUNCTION = "segment_interactive"
    CATEGORY = _category

    def segment_interactive(self, image: torch.Tensor, model_info: dict, points_json: str, 
                             output_category_mask: bool, output_confidence_mask: bool):
        """Performs interactive segmentation."""
        
        task_type = model_info.get('task_type')
        expected_task_type = MediaPipeInteractiveSegmenterModelLoaderNode.TASK_TYPE
        if not isinstance(model_info, dict) or task_type != expected_task_type:
             raise ValueError(f"Invalid model_info. Expected task_type '{expected_task_type}' but got '{task_type}'.")
        model_path = model_info.get('model_path')
        if not model_path:
             raise ValueError("Model path not found or invalid in model_info.")
            
        # Parse points from JSON
        try:
            points_list = json.loads(points_json)
            if not isinstance(points_list, list) or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in points_list):
                raise ValueError("Invalid format")
            points_of_interest = [PointOfInterest(x=p[0], y=p[1]) for p in points_list]
        except Exception as e:
            raise ValueError(f"Invalid points_json format: {e}. Expected list of [x, y] pairs, e.g., [[0.5, 0.5]].")

        # Manage detector instance
        if self._processor is None or self._model_path != model_path:
            if self._processor and hasattr(self._processor, 'close'): 
                 try: self._processor.close() 
                 except Exception as e: logger.warning(f"Error closing InteractiveSegmenterProcessor: {e}")
            logger.info(f"Creating new InteractiveSegmenterProcessor instance for {model_path}")
            self._processor = InteractiveSegmenterProcessor(model_path)
            self._model_path = model_path
             
        # Call the detector's segment method
        mask_batch = self._processor.segment(
            image,
            points_of_interest=points_of_interest,
            output_category_mask=output_category_mask,
            output_confidence_mask=output_confidence_mask
        )
        
        # Convert HW mask list to B1HW tensor for ComfyUI MASK type
        # Assuming detector returns one mask per batch item
        comfy_masks = []
        for mask_hw in mask_batch:
             if mask_hw is not None:
                  comfy_masks.append(mask_hw.unsqueeze(0)) # Add channel dim
             else:
                  # Handle None case - maybe return zero mask? Needs image shape
                  h, w = image.shape[1], image.shape[2]
                  comfy_masks.append(torch.zeros((1, h, w), dtype=torch.float32, device=image.device))
        
        output_tensor = torch.stack(comfy_masks, dim=0) # Stack along batch dim
        return (output_tensor,)

    def __del__(self):
         if hasattr(self, '_processor') and self._processor and hasattr(self._processor, 'close'):
             try: self._processor.close() 
             except Exception as e: logger.warning(f"Error closing InteractiveSegmenterProcessor in __del__: {e}")
             self._processor = None

# --- Mappings --- 
NODE_CLASS_MAPPINGS = {
    "MediaPipeInteractiveSegmenterModelLoader": MediaPipeInteractiveSegmenterModelLoaderNode,
    "MediaPipeInteractiveSegmenter": MediaPipeInteractiveSegmenterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeInteractiveSegmenterModelLoader": "Load Interactive Segmenter Model (MediaPipe)",
    "MediaPipeInteractiveSegmenter": "Interactive Segmenter (MediaPipe)",
} 