import torch
import logging

# Import Base Loader and Detector
from ..common.model_loader   import MediaPipeModelLoaderBaseNode
from ...src.face_stylization.detector import FaceStylizer

logger = logging.getLogger(__name__)
_category = "MediaPipeVision/Face/FaceStylization"
# --- Model Loader --- 
class MediaPipeFaceStylizerModelLoaderNode(MediaPipeModelLoaderBaseNode):
    """ComfyUI node for loading MediaPipe Face Stylizer models."""
    TASK_TYPE = "face_stylizer" # Need to add this task to MEDIAPIPE_MODELS
    RETURN_TYPES = ("FACE_STYLIZER_MODEL_INFO",)
    RETURN_NAMES = ("model_info",)
    CATEGORY = _category
    DESCRIPTION = "Loads a MediaPipe Face Stylizer model that can transform face images into artistic styles. Required before using the Face Stylizer node."
# --- Stylizer Node --- 
class MediaPipeFaceStylizerNode:
    """ComfyUI node for MediaPipe Face Stylization."""

    def __init__(self):
        self._detector: FaceStylizer = None
        self._model_path: str = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image containing faces to be artistically stylized"}),
                "model_info": ("FACE_STYLIZER_MODEL_INFO", {"tooltip": "Face stylizer model loaded from the 'Load Face Stylizer Model' node"}),
            }
        }

    RETURN_TYPES = ("IMAGE",) # Output is a stylized image
    RETURN_NAMES = ("stylized_image",)
    FUNCTION = "stylize_face"
    CATEGORY = _category
    DESCRIPTION = "Applies artistic styles to faces in images without affecting the background. Creates cartoon-like or artistic renditions of faces while preserving identity."

    def stylize_face(self, image: torch.Tensor, model_info: dict):
        """Applies face stylization."""
        
        task_type = model_info.get('task_type')
        expected_task_type = MediaPipeFaceStylizerModelLoaderNode.TASK_TYPE
        if not isinstance(model_info, dict) or task_type != expected_task_type:
             raise ValueError(f"Invalid model_info. Expected task_type '{expected_task_type}' but got '{task_type}'.")
        model_path = model_info.get('model_path')
        if not model_path:
             raise ValueError("Model path not found or invalid in model_info.")
            
        # Manage detector instance
        if self._detector is None or self._model_path != model_path:
            if self._detector and hasattr(self._detector, 'close'): 
                 try: self._detector.close() 
                 except Exception as e: logger.warning(f"Error closing FaceStylizer: {e}")
            logger.info(f"Creating new FaceStylizer instance for {model_path}")
            self._detector = FaceStylizer(model_path)
            self._model_path = model_path
             
        # Call the detector's stylize method
        stylized_image_batch = self._detector.stylize(image)
        
        return (stylized_image_batch,)

    def __del__(self):
         if hasattr(self, '_detector') and self._detector and hasattr(self._detector, 'close'):
             try: self._detector.close() 
             except Exception as e: logger.warning(f"Error closing FaceStylizer in __del__: {e}")
             self._detector = None

# --- Mappings --- 
NODE_CLASS_MAPPINGS = {
    "MediaPipeFaceStylizerModelLoader": MediaPipeFaceStylizerModelLoaderNode,
    "MediaPipeFaceStylizer": MediaPipeFaceStylizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeFaceStylizerModelLoader": "Load Face Stylizer Model (MediaPipe)",
    "MediaPipeFaceStylizer": "Face Stylizer (MediaPipe)",
} 