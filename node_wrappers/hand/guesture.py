import torch
import logging

# Import Base Loader and Detector
from ..common.model_loader import MediaPipeModelLoaderBaseNode
from ..common.base_detector_node import BaseMediaPipeDetectorNode
from ...src.gesture_recognition.detector import GestureRecognizer

logger = logging.getLogger(__name__)
_category = "MediaPipeVision/Hand/GestureRecognition"
# --- Model Loader --- 
class MediaPipeGestureRecognizerModelLoaderNode(MediaPipeModelLoaderBaseNode):
    """ComfyUI node for loading MediaPipe Gesture Recognizer models."""
    TASK_TYPE = "gesture_recognizer" # Need to add this task to MEDIAPIPE_MODELS in src/model_loader.py
    RETURN_TYPES = ("GESTURE_RECOGNIZER_MODEL_INFO",)
    RETURN_NAMES = ("model_info",)
    CATEGORY = _category
    DESCRIPTION = "Loads a MediaPipe Gesture Recognizer model for identifying hand gestures in images, such as thumbs up, open palm, pointing, etc. Required before using the Gesture Recognizer node."
# --- Recognizer Node --- 
class MediaPipeGestureRecognizerNode(BaseMediaPipeDetectorNode):
    """ComfyUI node for MediaPipe Gesture Recognition."""

    # Define class variables required by the base class
    DETECTOR_CLASS = GestureRecognizer
    MODEL_INFO_TYPE = "GESTURE_RECOGNIZER_MODEL_INFO"
    EXPECTED_TASK_TYPE = "gesture_recognizer"
    RETURN_TYPES = ("GESTURE_RECOGNITIONS",)
    RETURN_NAMES = ("gesture_recognitions",)
    FUNCTION = "recognize_gestures"
    CATEGORY = _category
    DESCRIPTION = "Identifies common hand gestures in images like thumbs up, victory sign, open palm, etc. Provides both gesture names and confidence scores for recognized hand poses."

    @classmethod
    def INPUT_TYPES(cls):
        # Start with the base inputs from the parent class
        inputs = super().INPUT_TYPES()
        
        # Add gesture-specific parameters
        inputs["required"].update({
            "max_results": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1,
                                    "tooltip": "Maximum number of hands to analyze in the image - higher values can detect more hands but use more processing power"}),
            "min_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Minimum confidence threshold for hand detection - lower values detect more hands but may include false positives"}),
            "min_tracking_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "(VIDEO mode) Minimum confidence threshold for tracking hands between frames - lower values maintain tracking longer but may be less accurate"}),
            "min_presence_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "(VIDEO mode) Minimum confidence that a hand is present - lower values may detect hands that are less clear"}),
        })
        
        return inputs

    def recognize_gestures(self, image: torch.Tensor, model_info: dict, max_results: int, 
                           min_confidence: float, min_tracking_confidence: float, min_presence_confidence: float, 
                           running_mode: str, delegate: str):
        """Performs gesture recognition on the input image."""
        
        # Validate model_info and get model path
        model_path = self.validate_model_info(model_info)
        
        # Initialize or update detector
        detector = self.initialize_or_update_detector(model_path)
        
        # Call the detector's recognize method with standardized parameters
        gesture_results_batch = detector.recognize(
            image,
            num_hands=max_results,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_presence_confidence=min_presence_confidence,
            running_mode=running_mode,
            delegate=delegate
        )
        
        return (gesture_results_batch,)

# --- Mappings --- 
NODE_CLASS_MAPPINGS = {
    "MediaPipeGestureRecognizerModelLoader": MediaPipeGestureRecognizerModelLoaderNode,
    "MediaPipeGestureRecognizer": MediaPipeGestureRecognizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeGestureRecognizerModelLoader": "Load Gesture Recognizer Model (MediaPipe)",
    "MediaPipeGestureRecognizer": "Gesture Recognizer (MediaPipe)",
} 