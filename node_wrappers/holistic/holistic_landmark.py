import torch
import cv2
import mediapipe as mp
import numpy as np
from ...src.holistic_landmark.detector import HolisticLandmarkDetector
from ..common.base_detector_node import BaseMediaPipeDetectorNode
import logging

logger = logging.getLogger(__name__)
_category = "MediaPipeVision/Holistic"

class MediaPipeHolisticLandmarkerNode(BaseMediaPipeDetectorNode):
    """ComfyUI node for MediaPipe Holistic Landmark Detection.
    
    This node uses the legacy MediaPipe Holistic API that combines face, pose and hand
    tracking in a single detector.
    
    Note: This uses the legacy MediaPipe Python Solution API while we wait for an official
    Holistic Task implementation, and thus it stands out among other implementations in this project. 
    It does not require a model file to be loaded separately.
    """

    # Define class variables required by the base class
    DETECTOR_CLASS = HolisticLandmarkDetector
    MODEL_INFO_TYPE = None  # Not used - legacy solution doesn't need model file
    EXPECTED_TASK_TYPE = None  # Not used - legacy solution doesn't need model info
    RETURN_TYPES = ("HOLISTIC_LANDMARKS", "MASK_LIST",)
    RETURN_NAMES = ("landmarks", "segmentation_masks")
    FUNCTION = "detect_holistic"
    CATEGORY = _category
    DESCRIPTION = """
    LEGACY MODEL ***USE CAUTION***: 
    RUNNING THIS DETECTOR WILL BREAK ALL OTHER DETECTORS THAT USE THE TASK API UNTIL COMFYUI IS RESTARTED.
    Detects face, pose, and hand landmarks in a unified detection. Provides comprehensive tracking of an entire person. This uses the legacy MediaPipe API rather than the newer Task API.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define inputs for holistic detection."""
        # We override the parent INPUT_TYPES completely since we don't need model_info
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image for detection"}),
                "min_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                 "tooltip": "Minimum confidence threshold for detection - lower values detect more poses but may include false positives"}),
                "min_tracking_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Minimum confidence threshold for tracking - lower values maintain tracking longer but may be less accurate"}),
                "model_complexity": ("INT", {"default": 1, "min": 0, "max": 2, "step": 1,
                                  "tooltip": "Model complexity (0=Lite, 1=Full, 2=Heavy) - higher values are more accurate but slower"}),
                "enable_segmentation": ("BOOLEAN", {"default": False,
                                       "tooltip": "When enabled, generates masks that separate people from the background"}),
                "refine_face_landmarks": ("BOOLEAN", {"default": False, 
                                        "tooltip": "When enabled, provides more detailed face landmarks around eyes and lips"}),
                "static_image_mode": ("BOOLEAN", {"default": False,
                                    "tooltip": "When enabled, treats each frame as independent - more accurate but slower and doesn't benefit from tracking"}),
            }
        }

    def detect_holistic(self, image: torch.Tensor, 
                     min_confidence: float, 
                     min_tracking_confidence: float, 
                     model_complexity: int,
                     enable_segmentation: bool, 
                     refine_face_landmarks: bool,
                     static_image_mode: bool):
        """Performs holistic landmark detection with the configured parameters.
        
        Unlike other detectors in this codebase, this doesn't need a model_info parameter
        as it uses the built-in legacy Holistic API.
        """
        # Initialize detector (without model path)
        detector = self.initialize_or_update_detector(None)
        
        # Perform detection with all parameters
        holistic_landmarks_batch, segmentation_masks_batch = detector.detect(
            image,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            refine_face_landmarks=refine_face_landmarks,
            static_image_mode=static_image_mode
        )
        
        return (holistic_landmarks_batch, segmentation_masks_batch or [])
    
    def initialize_or_update_detector(self, model_path=None):
        """Override to handle the case where model_path is not needed."""
        if self._detector is None:
            self._detector = self.DETECTOR_CLASS()
        return self._detector
    
    def validate_model_info(self, model_info=None):
        """Override to handle the case where model_info is not needed."""
        # No validation needed for legacy API
        return None


# Define mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MediaPipeHolisticLandmarker": MediaPipeHolisticLandmarkerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeHolisticLandmarker": "**CAUTION** Holistic Landmarker (MediaPipe Legacy)",
} 