import torch
from ...src.face_detection.detector import FaceDetector
from ..common.model_loader import MediaPipeModelLoaderBaseNode
from ..common.base_detector_node import BaseMediaPipeDetectorNode
import logging

logger = logging.getLogger(__name__)
_category = "MediaPipeVision/Face/FaceDetection"
# Inherit from Base Loader
class MediaPipeFaceDetectorModelLoaderNode(MediaPipeModelLoaderBaseNode):
    """ComfyUI node for loading MediaPipe Face Detector models."""
    TASK_TYPE = "face_detector"
    RETURN_TYPES = ("FACE_DETECTOR_MODEL_INFO",) # Specific return type
    RETURN_NAMES = ("model_info",)
    CATEGORY = _category
    DESCRIPTION = "Loads a MediaPipe Face Detector model for finding faces in images. Required before using the Face Detector node."
    # INPUT_TYPES and FUNCTION inherited

# Detection/Landmarking Nodes
class MediaPipeFaceDetectorNode(BaseMediaPipeDetectorNode):
    """ComfyUI node for MediaPipe Face Detection."""
    
    # Define class variables required by the base class
    DETECTOR_CLASS = FaceDetector
    MODEL_INFO_TYPE = "FACE_DETECTOR_MODEL_INFO"
    EXPECTED_TASK_TYPE = "face_detector"
    RETURN_TYPES = ("FACE_DETECTIONS",)
    RETURN_NAMES = ("face_detections",)
    FUNCTION = "detect"
    CATEGORY = _category
    DESCRIPTION = "Detects faces in images using MediaPipe's face detection technology. Provides coordinates and confidence scores for all faces found in the image."
        
    @classmethod
    def INPUT_TYPES(cls):
        # Start with the base inputs from the parent class
        inputs = super().INPUT_TYPES()
        
        # Add face detection specific parameters
        inputs["required"].update({
            "min_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                         "tooltip": "Minimum confidence threshold for face detection - lower values find more faces but may include false positives"}),
        })
        
        return inputs

    def detect(self, image: torch.Tensor, model_info: dict, min_confidence: float, 
               running_mode: str, delegate: str):
        """Performs face detection with the configured parameters."""
        
        # Validate model_info and get model path
        model_path = self.validate_model_info(model_info)
        
        # Initialize or update detector
        detector = self.initialize_or_update_detector(model_path)
        
        # Perform detection with all parameters
        batch_results = detector.detect(
            image,
            min_detection_confidence=min_confidence,
            running_mode=running_mode,
            delegate=delegate
        )
        
        return (batch_results,)

# Define mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MediaPipeFaceDetectorModelLoader": MediaPipeFaceDetectorModelLoaderNode,
    "MediaPipeFaceDetector": MediaPipeFaceDetectorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeFaceDetectorModelLoader": "Load Face Detector Model (MediaPipe)",
    "MediaPipeFaceDetector": "Face Detector (MediaPipe)",
}
