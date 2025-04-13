import torch
from ...src.object_detection.detector import ObjectDetector
from ..common.model_loader import MediaPipeModelLoaderBaseNode
from ..common.base_detector_node import BaseMediaPipeDetectorNode
import logging

logger = logging.getLogger(__name__)
_category = "MediaPipeVision/ObjectDetection"

class MediaPipeObjectDetectorModelLoaderNode(MediaPipeModelLoaderBaseNode):
    """ComfyUI node for loading MediaPipe Object Detector models."""
    TASK_TYPE = "object_detector"
    RETURN_TYPES = ("OBJECT_DETECTOR_MODEL_INFO",)
    RETURN_NAMES = ("model_info",)
    CATEGORY = _category

class MediaPipeObjectDetectorNode(BaseMediaPipeDetectorNode):
    """ComfyUI node for MediaPipe Object Detection."""

    # Define class variables required by the base class
    DETECTOR_CLASS = ObjectDetector
    MODEL_INFO_TYPE = "OBJECT_DETECTOR_MODEL_INFO"
    EXPECTED_TASK_TYPE = "object_detector"
    RETURN_TYPES = ("OBJECT_DETECTIONS",)
    RETURN_NAMES = ("object_detections",)
    FUNCTION = "detect"
    CATEGORY = _category

    @classmethod
    def INPUT_TYPES(cls):
        # Start with the base inputs from the parent class
        inputs = super().INPUT_TYPES()
        
        # Add object detection specific parameters
        inputs["required"].update({
            "min_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Minimum confidence score for detected objects"}),
            "max_results": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1,
                                 "tooltip": "Maximum number of objects to detect"}),
        })
        
        return inputs

    def detect(self, image: torch.Tensor, model_info: dict, min_confidence: float, 
               max_results: int, running_mode: str, delegate: str):
        """Performs object detection with the configured parameters."""
        
        # Validate model_info and get model path
        model_path = self.validate_model_info(model_info)
        
        # Initialize or update detector
        detector = self.initialize_or_update_detector(model_path)
        
        # Perform detection with all parameters
        batch_results = detector.detect(
            image,
            score_threshold=min_confidence,
            max_results=max_results,
            running_mode=running_mode,
            delegate=delegate
        )
        
        return (batch_results,)

# Define mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MediaPipeObjectDetectorModelLoader": MediaPipeObjectDetectorModelLoaderNode,
    "MediaPipeObjectDetector": MediaPipeObjectDetectorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeObjectDetectorModelLoader": "Load Object Detector Model (MediaPipe)",
    "MediaPipeObjectDetector": "Object Detector (MediaPipe)",
} 