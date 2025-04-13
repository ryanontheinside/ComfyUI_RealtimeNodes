# node_wrappers/face/masking.py
import torch
import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple

from ...src.types import LandmarkPoint
from ...src.utils.mask_utils import create_mask_from_points

logger = logging.getLogger(__name__)
_category = "MediaPipeVision/Face/FaceLandmark/Masking"
# --- Face Part Definitions --- 
FACE_PART_INDICES = { # NOTE: These indices are EXAMPLES and NEED VERIFICATION!
    "FACE_OVAL": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "LIPS_OUTER": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 62, 185, 40, 39, 37, 0, 267, 269, 270, 409, 292],
    "LIPS_INNER": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 76, 191, 81, 82, 13, 312, 311, 310, 415, 306],
    "LEFT_EYE": [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173],
    "RIGHT_EYE": [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398],
    "LEFT_EYEBROW": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "RIGHT_EYEBROW": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    "NOSE_BRIDGE": [168, 6, 197, 195, 5],
    "NOSE_TIP_LOWER": [5, 4, 1, 19, 94, 2],
}

# --- Node Definition --- 
class MaskFromFaceLandmarksNode:
    """Creates a mask for a specific facial part from Face Landmarks."""
    CATEGORY = _category
    DESCRIPTION = "Creates precise masks for specific facial features like eyes, lips, nose, or the entire face. Perfect for targeted editing, inpainting, or creating effects on specific facial regions."

    @classmethod
    def INPUT_TYPES(cls):
        part_names = list(FACE_PART_INDICES.keys())
        return {
            "required": {
                "face_landmarks": ("FACE_LANDMARKS", {"tooltip": "Facial landmark data from the Face Landmarker node containing the precise locations of facial features"}),
                "image_for_dimensions": ("IMAGE", {"tooltip": "Reference image to determine the size of the output mask - the mask will match these dimensions"}),
                "part_name": (part_names, {"default": "FACE_OVAL", 
                                         "tooltip": "Choose which facial feature to create a mask for - FACE_OVAL (whole face), LIPS_OUTER (mouth), LEFT_EYE, etc."}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "generate_mask"

    def generate_mask(self, face_landmarks: List[List[List[LandmarkPoint]]], image_for_dimensions: torch.Tensor, part_name: str):
        if part_name not in FACE_PART_INDICES:
            raise ValueError(f"Unknown face part name: '{part_name}'. Available: {list(FACE_PART_INDICES.keys())}")
        
        # Get dimensions from image
        if image_for_dimensions.dim() != 4:
            raise ValueError("image_for_dimensions must be BHWC format.")
        height = image_for_dimensions.shape[1]
        width = image_for_dimensions.shape[2]
        input_batch_size = image_for_dimensions.shape[0]
        device = image_for_dimensions.device

        # Validate batch sizes match
        if len(face_landmarks) != input_batch_size:
            logger.warning(f"Batch size mismatch between face_landmarks ({len(face_landmarks)}) and image_for_dimensions ({input_batch_size}). Using landmark batch size.")
            # Adjust effective batch size based on landmarks if mismatched, though ideally they match
            batch_size = len(face_landmarks)
        else:
             batch_size = input_batch_size
        
        required_indices = FACE_PART_INDICES[part_name]
        output_masks = []

        for i in range(batch_size):
            landmarks_for_image = face_landmarks[i] # List[List[LandmarkPoint]] (Faces in image)
            combined_points_for_image = []

            # Combine points from all detected faces for the selected part
            for face_result_landmarks in landmarks_for_image: # face_result_landmarks is List[LandmarkPoint]
                points_dict = {lm.index: lm for lm in face_result_landmarks} # Iterate directly over the list
                part_points = []
                for idx in required_indices:
                    if idx in points_dict:
                        lm = points_dict[idx]
                        px = int(lm.x * width)
                        py = int(lm.y * height)
                        part_points.append((px, py))
                    # Removed warning here to avoid spam if landmarks are just missing
                combined_points_for_image.extend(part_points)
            
            # Create mask from all collected points for this image
            mask_tensor = create_mask_from_points(height, width, combined_points_for_image, device=device)
            output_masks.append(mask_tensor)

        if not output_masks:
             logger.warning("No landmarks found to generate any masks. Returning zero mask batch.")
             output_batch = torch.zeros((input_batch_size, 1, height, width), dtype=torch.float32, device=device)
        else:
             # Stack along batch dimension, ensure channel dim is present
             output_batch = torch.stack(output_masks, dim=0) # Shape becomes (B, 1, H, W)
             
        return (output_batch,)

# Remove GetImageDimensionsNode class definition

# --- Mappings --- 
NODE_CLASS_MAPPINGS = {
    "MaskFromFaceLandmarks": MaskFromFaceLandmarksNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFromFaceLandmarks": "Mask From Face Landmarks (MediaPipe)",
} 