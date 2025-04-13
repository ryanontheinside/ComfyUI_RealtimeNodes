import torch
from ...src.hand_landmark.detector import HandLandmarkDetector
from ..common.model_loader import MediaPipeModelLoaderBaseNode
from ..common.base_detector_node import BaseMediaPipeDetectorNode
import logging

logger = logging.getLogger(__name__)

_category = "MediaPipeVision/Hand/HandLandmark"
class MediaPipeHandLandmarkerModelLoaderNode(MediaPipeModelLoaderBaseNode):
    """ComfyUI node for loading MediaPipe Hand Landmarker models."""
    TASK_TYPE = "hand_landmarker"
    RETURN_TYPES = ("HAND_LANDMARKER_MODEL_INFO",)
    RETURN_NAMES = ("model_info",)
    CATEGORY = _category
    DESCRIPTION = "Loads a MediaPipe Hand Landmarker model for detecting hand positions and gestures in images. Required before using the Hand Landmarker node."
    
class MediaPipeHandLandmarkerNode(BaseMediaPipeDetectorNode):
    """ComfyUI node for MediaPipe Hand Landmark Detection."""

    # Define class variables required by the base class
    DETECTOR_CLASS = HandLandmarkDetector
    MODEL_INFO_TYPE = "HAND_LANDMARKER_MODEL_INFO"
    EXPECTED_TASK_TYPE = "hand_landmarker"
    RETURN_TYPES = ("HAND_LANDMARKS", "HANDEDNESS_LIST")
    RETURN_NAMES = ("landmarks", "handedness")
    FUNCTION = "detect"
    CATEGORY = _category
    DESCRIPTION = "Detects hands in images and provides precise landmark points for each finger joint and hand position. Also identifies whether each hand is left or right."

    @classmethod
    def INPUT_TYPES(cls):
        # Start with the base inputs from the parent class
        inputs = super().INPUT_TYPES()
        
        # Add hand-specific parameters
        inputs["required"].update({
            "max_results": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1,
                                "tooltip": "Maximum number of hands to detect and track in the image - higher values can detect more hands but use more processing power"}),
            "min_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Minimum confidence threshold for hand detection - lower values detect more hands but may include false positives"}),
            "min_presence_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "(VIDEO mode) Minimum confidence that a hand is present - lower values may detect hands that are less clear"}),
            "min_tracking_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "(VIDEO mode) Minimum confidence for tracking a hand between frames - lower values maintain tracking longer but may drift"}),
        })
        
        return inputs

    def detect(self, image: torch.Tensor, model_info: dict, max_results: int, 
               min_confidence: float, min_presence_confidence: float, 
               min_tracking_confidence: float, running_mode: str, delegate: str):
        """Performs hand landmark detection with the configured parameters."""
        
        # Validate model_info and get model path
        model_path = self.validate_model_info(model_info)
        
        # Initialize or update detector
        detector = self.initialize_or_update_detector(model_path)
        
        # Perform detection with all parameters
        hand_landmarks_batch = detector.detect(
            image,
            num_hands=max_results,
            min_detection_confidence=min_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=running_mode,
            delegate=delegate
        )
        
        # Extract handedness information
        handedness_batch = []
        if hand_landmarks_batch:
            for image_hands in hand_landmarks_batch:
                handedness_per_image = [hand.handedness for hand in image_hands if hand.handedness is not None]
                handedness_batch.append(handedness_per_image)
        
        return (hand_landmarks_batch, handedness_batch)

NODE_CLASS_MAPPINGS = {
    "MediaPipeHandLandmarkerModelLoader": MediaPipeHandLandmarkerModelLoaderNode,
    "MediaPipeHandLandmarker": MediaPipeHandLandmarkerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeHandLandmarkerModelLoader": "Load Hand Landmarker Model (MediaPipe)",
    "MediaPipeHandLandmarker": "Hand Landmarker (MediaPipe)",
} 