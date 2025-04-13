# node_wrappers/hand/masking.py
import torch
import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple

from ...src.types import HandLandmarksResult
# Reusing the helper function from face masking
from ...src.utils.mask_utils import create_mask_from_points 

logger = logging.getLogger(__name__)
_category = "MediaPipeVision/Hand/HandLandmark/Masking"
# --- Hand Part Definitions (Based on standard MediaPipe Hand Indices) ---
# These indices need verification
HAND_PART_INDICES = {
    "PALM": [0, 1, 5, 9, 13, 17], # Example
    "THUMB": [1, 2, 3, 4], # Example
    "INDEX_FINGER": [5, 6, 7, 8], # Example
    "MIDDLE_FINGER": [9, 10, 11, 12], # Example
    "RING_FINGER": [13, 14, 15, 16], # Example
    "PINKY_FINGER": [17, 18, 19, 20], # Example
    "WRIST": [0], # Single point, convex hull won't work
    "ALL_FINGERS": list(range(1, 21)), # Example excluding wrist
    "FULL_HAND": list(range(21)), # All points
}

# --- Node Definition --- 
class MaskFromHandLandmarksNode:
    """Creates a mask for a specific hand part from Hand Landmarks."""
    CATEGORY = _category
    DESCRIPTION = "Creates precise masks for specific hand parts like individual fingers, palm, or the entire hand. Perfect for targeted editing or creating effects on hands and fingers."

    @classmethod
    def INPUT_TYPES(cls):
        part_names = list(HAND_PART_INDICES.keys())
        return {
            "required": {
                "hand_landmarks": ("HAND_LANDMARKS", {"tooltip": "Hand landmark data from the Hand Landmarker node containing the precise locations of hand and finger joints"}),
                "image_for_dimensions": ("IMAGE", {"tooltip": "Reference image to determine the size of the output mask - the mask will match these dimensions"}),
                "part_name": (part_names, {"default": "FULL_HAND", 
                                        "tooltip": "Choose which part of the hand to create a mask for - FULL_HAND (entire hand), THUMB, INDEX_FINGER, PALM, etc."}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "generate_mask"

    def generate_mask(self, hand_landmarks: List[List[HandLandmarksResult]], image_for_dimensions: torch.Tensor, part_name: str):
        if part_name not in HAND_PART_INDICES:
            raise ValueError(f"Unknown hand part name: '{part_name}'. Available: {list(HAND_PART_INDICES.keys())}")
        
        # Get dimensions from image
        if image_for_dimensions.dim() != 4:
            raise ValueError("image_for_dimensions must be BHWC format.")
        height = image_for_dimensions.shape[1]
        width = image_for_dimensions.shape[2]
        input_batch_size = image_for_dimensions.shape[0]
        device = image_for_dimensions.device

        # Validate batch sizes match
        if len(hand_landmarks) != input_batch_size:
             logger.warning(f"Batch size mismatch between hand_landmarks ({len(hand_landmarks)}) and image_for_dimensions ({input_batch_size}). Using landmark batch size.")
             batch_size = len(hand_landmarks)
        else:
             batch_size = input_batch_size

        required_indices = HAND_PART_INDICES[part_name]
        output_masks = []

        for i in range(batch_size):
            landmarks_for_image = hand_landmarks[i] # List[HandLandmarksResult]
            combined_points_for_image = []

            # Combine points from all detected hands for the selected part
            for hand_result in landmarks_for_image:
                points_dict = {lm.index: lm for lm in hand_result.landmarks}
                part_points = []
                for idx in required_indices:
                    if idx in points_dict:
                        lm = points_dict[idx]
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
    "MaskFromHandLandmarks": MaskFromHandLandmarksNode,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFromHandLandmarks": "Mask From Hand Landmarks (MediaPipe)",
} 