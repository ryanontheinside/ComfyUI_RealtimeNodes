"""Node wrapper for face detection."""

import torch
from ...src.face_landmark.detector import FaceLandmarkDetector
# Import Base Classes
from ..common.model_loader import MediaPipeModelLoaderBaseNode
from ..common.base_detector_node import BaseMediaPipeDetectorNode
import logging

logger = logging.getLogger(__name__)

_category = "MediaPipeVision/Face/FaceLandmark"
# Inherit from Base Loader
class MediaPipeFaceLandmarkerModelLoaderNode(MediaPipeModelLoaderBaseNode):
    """ComfyUI node for loading MediaPipe Face Landmarker models."""
    TASK_TYPE = "face_landmarker"
    RETURN_TYPES = ("FACE_LANDMARKER_MODEL_INFO",) # Specific return type
    RETURN_NAMES = ("model_info",)
    CATEGORY = _category
    DESCRIPTION = "Loads a MediaPipe Face Landmarker model which detects facial landmarks, expressions and face orientation. Required before using the Face Landmarker node."
    # INPUT_TYPES and FUNCTION inherited


class MediaPipeFaceLandmarkerNode(BaseMediaPipeDetectorNode):
    """ComfyUI node for MediaPipe Face Landmark Detection."""

    # Define class variables required by the base class
    DETECTOR_CLASS = FaceLandmarkDetector
    MODEL_INFO_TYPE = "FACE_LANDMARKER_MODEL_INFO"
    EXPECTED_TASK_TYPE = "face_landmarker"
    RETURN_TYPES = ("FACE_LANDMARKS", "BLENDSHAPES_LIST", "TRANSFORM_MATRIX_LIST")
    RETURN_NAMES = ("landmarks", "blendshapes", "transform_matrices")
    FUNCTION = "detect"
    CATEGORY = _category
    DESCRIPTION = "Detects precise facial landmarks (478 points), expressions, and head orientation in images. Provides detailed data for face tracking, animation, or specialized face image processing."

    @classmethod
    def INPUT_TYPES(cls):
        # Start with the base inputs from the parent class
        inputs = super().INPUT_TYPES()
        
        # Add face-specific parameters
        inputs["required"].update({
            "max_results": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
                               "tooltip": "Maximum number of faces to analyze - higher values detect landmarks on more faces but use more processing power"}),
            "min_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                                "tooltip": "Minimum confidence threshold for detecting faces - lower finds more faces but may include false positives"}),
            "min_presence_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                               "tooltip": "(VIDEO mode) Minimum confidence that a face is present - lower values may detect faces that are less clear"}),
            "min_tracking_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "(VIDEO mode) Minimum confidence for tracking a face between frames - lower values maintain tracking longer but may drift"}),
            "output_blendshapes": ("BOOLEAN", {"default": False,
                                       "tooltip": "When enabled, outputs facial expression data (smile, frown, etc.) useful for animation or emotion analysis"}),
            "output_transform_matrix": ("BOOLEAN", {"default": False,
                                            "tooltip": "When enabled, outputs 3D face orientation data (head pose) useful for AR/VR applications"}),
        })
        
        return inputs

    def detect(self, image: torch.Tensor, model_info: dict, max_results: int,
            min_confidence: float, min_presence_confidence: float,
            min_tracking_confidence: float, output_blendshapes: bool, 
            output_transform_matrix: bool, running_mode: str, delegate: str):
        """Performs face landmark detection with the configured parameters."""
        
        # Validate model_info and get model path
        model_path = self.validate_model_info(model_info)
        
        # Initialize or update detector
        detector = self.initialize_or_update_detector(model_path)
        
        # Perform detection with all parameters
        face_landmarks_batch, blendshapes_batch, matrices_batch = detector.detect(
            image,
            num_faces=max_results,
            min_detection_confidence=min_confidence,
            min_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_blendshapes=output_blendshapes,
            output_transform_matrix=output_transform_matrix,
            running_mode=running_mode,
            delegate=delegate
        )
        
        # Return the results with optional empty lists for disabled outputs
        return (
            face_landmarks_batch, 
            blendshapes_batch if output_blendshapes else [], 
            matrices_batch if output_transform_matrix else []
        )

# Define mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MediaPipeFaceLandmarkerModelLoader": MediaPipeFaceLandmarkerModelLoaderNode,
    "MediaPipeFaceLandmarker": MediaPipeFaceLandmarkerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeFaceLandmarkerModelLoader": "Load Face Landmarker Model (MediaPipe)",
    "MediaPipeFaceLandmarker": "Face Landmarker (MediaPipe)",
} 