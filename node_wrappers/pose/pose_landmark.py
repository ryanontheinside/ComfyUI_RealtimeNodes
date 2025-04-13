import torch
from ...src.pose_landmark.detector import PoseLandmarkDetector
from ..common.model_loader import MediaPipeModelLoaderBaseNode
from ..common.base_detector_node import BaseMediaPipeDetectorNode
import logging

logger = logging.getLogger(__name__)
_category = "MediaPipeVision/Pose"

class MediaPipePoseLandmarkerModelLoaderNode(MediaPipeModelLoaderBaseNode):
    """ComfyUI node for loading MediaPipe Pose Landmarker models."""
    TASK_TYPE = "pose_landmarker"
    RETURN_TYPES = ("POSE_LANDMARKER_MODEL_INFO",)
    RETURN_NAMES = ("model_info",)
    CATEGORY = _category
    DESCRIPTION = "Loads a MediaPipe Pose Landmarker model for detecting human body poses and postures in images. Required before using the Pose Landmarker node."

class MediaPipePoseLandmarkerNode(BaseMediaPipeDetectorNode):
    """ComfyUI node for MediaPipe Pose Landmark Detection."""

    # Define class variables required by the base class
    DETECTOR_CLASS = PoseLandmarkDetector
    MODEL_INFO_TYPE = "POSE_LANDMARKER_MODEL_INFO"
    EXPECTED_TASK_TYPE = "pose_landmarker"
    RETURN_TYPES = ("POSE_LANDMARKS", "MASK_LIST",)
    RETURN_NAMES = ("landmarks", "segmentation_masks")
    FUNCTION = "detect"
    CATEGORY = _category
    DESCRIPTION = "Detects human body poses in images and provides precise landmark points for major body joints and posture. Can also generate segmentation masks that isolate people from the background."

    @classmethod
    def INPUT_TYPES(cls):
        # Start with the base inputs from the parent class
        inputs = super().INPUT_TYPES()
        
        # Add pose-specific parameters
        inputs["required"].update({
            "max_results": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1,
                               "tooltip": "Maximum number of people/poses to detect in the image - higher values can detect more people but use more processing power"}),
            "min_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "Minimum confidence threshold for pose detection - lower values detect more poses but may include false positives"}),
            "min_tracking_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "(VIDEO mode) Minimum confidence threshold for pose tracking between frames - lower values maintain tracking longer but may be less accurate"}),
            "min_presence_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "(VIDEO mode) Minimum confidence threshold for pose presence in the image - lower values may include false negatives"}),
            "output_segmentation_masks": ("BOOLEAN", {"default": False,
                                          "tooltip": "When enabled, generates masks that separate people from the background"}),
        })
        
        return inputs

    def detect(self, image: torch.Tensor, model_info: dict, max_results: int, 
               min_confidence: float, min_tracking_confidence: float, 
               min_presence_confidence: float, output_segmentation_masks: bool, 
               running_mode: str, delegate: str):
        """Performs pose landmark detection with the configured parameters."""
        
        # Validate model_info and get model path
        model_path = self.validate_model_info(model_info)
        
        # Initialize or update detector
        detector = self.initialize_or_update_detector(model_path)
        
        # Perform detection with all parameters
        pose_landmarks_batch, segmentation_masks_batch = detector.detect(
            image,
            num_poses=max_results,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_presence_confidence=min_presence_confidence,
            output_segmentation_masks=output_segmentation_masks,
            running_mode=running_mode,
            delegate=delegate
        )
        
        # Process segmentation masks if needed
        flat_masks = []
        if segmentation_masks_batch:
            for img_masks in segmentation_masks_batch:
                if img_masks:
                    flat_masks.extend(img_masks)
        
        return (pose_landmarks_batch, flat_masks if output_segmentation_masks else [])

# Define mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MediaPipePoseLandmarkerModelLoader": MediaPipePoseLandmarkerModelLoaderNode,
    "MediaPipePoseLandmarker": MediaPipePoseLandmarkerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipePoseLandmarkerModelLoader": "Load Pose Landmarker Model (MediaPipe)",
    "MediaPipePoseLandmarker": "Pose Landmarker (MediaPipe)",
} 