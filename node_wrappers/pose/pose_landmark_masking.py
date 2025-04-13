# node_wrappers/pose/masking.py
import torch
import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple

from ...src.types import PoseLandmarksResult
from ...src.utils.mask_utils import create_mask_from_points  
# Removed GetImageDimensionsNode import

logger = logging.getLogger(__name__)
_category = "MediaPipeVision/Pose/Masking"

# --- Pose Part Definitions (Based on standard MediaPipe Pose Indices - BlazePose) ---
# These indices need verification
POSE_PART_INDICES = {
    "HEAD": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # Example
    "TORSO": [11, 12, 23, 24], # Example
    "LEFT_ARM": [11, 13, 15], # Upper arm only
    "RIGHT_ARM": [12, 14, 16], # Upper arm only
    "LEFT_FOREARM": [13, 15, 17, 19, 21], # Includes hand points?
    "RIGHT_FOREARM": [14, 16, 18, 20, 22], # Includes hand points?
    "LEFT_LEG": [23, 25, 27], # Upper leg
    "RIGHT_LEG": [24, 26, 28], # Upper leg
    "LEFT_LOWER_LEG": [25, 27, 29, 31], # Includes foot?
    "RIGHT_LOWER_LEG": [26, 28, 30, 32], # Includes foot?
    "FULL_BODY": list(range(33)),
}

# --- Node Definition --- 
class MaskFromPoseLandmarksNode:
    """Creates a mask for a specific body part from Pose Landmarks."""
    CATEGORY = _category
    DESCRIPTION = "Creates precise masks for specific body parts like head, torso, arms, or legs. Perfect for targeted editing or creating effects on specific body regions."

    @classmethod
    def INPUT_TYPES(cls):
        part_names = list(POSE_PART_INDICES.keys())
        return {
            "required": {
                "pose_landmarks": ("POSE_LANDMARKS", {"tooltip": "Pose landmark data from the Pose Landmarker node containing the precise locations of body joints"}),
                "image_for_dimensions": ("IMAGE", {"tooltip": "Reference image to determine the size of the output mask - the mask will match these dimensions"}),
                "part_name": (part_names, {"default": "FULL_BODY", 
                                         "tooltip": "Choose which body part to create a mask for - FULL_BODY (entire person), HEAD, TORSO, LEFT_ARM, etc."}),
                "min_visibility": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Only include landmarks that are at least this visible (0.0 = include all points regardless of visibility, 1.0 = only fully visible points)"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "generate_mask"

    def generate_mask(self, pose_landmarks: List[List[PoseLandmarksResult]], image_for_dimensions: torch.Tensor, part_name: str, min_visibility: float):
        if part_name not in POSE_PART_INDICES:
            raise ValueError(f"Unknown pose part name: '{part_name}'. Available: {list(POSE_PART_INDICES.keys())}")
        
        # Get dimensions from image
        if image_for_dimensions.dim() != 4:
            raise ValueError("image_for_dimensions must be BHWC format.")
        height = image_for_dimensions.shape[1]
        width = image_for_dimensions.shape[2]
        input_batch_size = image_for_dimensions.shape[0]
        device = image_for_dimensions.device

        # Validate batch sizes match
        if len(pose_landmarks) != input_batch_size:
            logger.warning(f"Batch size mismatch between pose_landmarks ({len(pose_landmarks)}) and image_for_dimensions ({input_batch_size}). Using landmark batch size.")
            batch_size = len(pose_landmarks)
        else:
            batch_size = input_batch_size

        required_indices = POSE_PART_INDICES[part_name]
        output_masks = []

        for i in range(batch_size):
            landmarks_for_image = pose_landmarks[i] # List[PoseLandmarksResult]
            combined_points_for_image = []

            # Combine points from all detected poses for the selected part
            for pose_result in landmarks_for_image:
                # Pose result already contains the .landmarks attribute
                points_dict = {lm.index: lm for lm in pose_result.landmarks}
                part_points = []
                for idx in required_indices:
                    if idx in points_dict:
                        lm = points_dict[idx]
                        is_visible = lm.visibility if lm.visibility is not None else 1.0
                        if is_visible >= min_visibility:
                            px = int(lm.x * width)
                            py = int(lm.y * height)
                            part_points.append((px, py))
                    # No warning for missing index
                combined_points_for_image.extend(part_points)
            
            # Create mask from all collected points for this image
            mask_tensor = create_mask_from_points(height, width, combined_points_for_image, device=device)
            output_masks.append(mask_tensor)

        if not output_masks:
             logger.warning("No landmarks found to generate any masks. Returning zero mask batch.")
             output_batch = torch.zeros((input_batch_size, 1, height, width), dtype=torch.float32, device=device)
        else:
             output_batch = torch.stack(output_masks, dim=0)
             
        return (output_batch,)

# --- Mappings --- 
NODE_CLASS_MAPPINGS = {
    "MaskFromPoseLandmarks": MaskFromPoseLandmarksNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFromPoseLandmarks": "Mask From Pose Landmarks (MediaPipe)",
} 