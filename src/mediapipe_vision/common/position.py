import logging
from typing import Optional, List, Any, Tuple, Union
import math
import torch
import numpy as np
import cv2
import collections
import typing # Add typing for Dict hint

#TODO: consider moving base nodes to src

# Assuming types are correctly imported relative to this file's location
from ..types import (
    FACE_LANDMARKS, HAND_LANDMARKS, POSE_LANDMARKS,
    LandmarkPoint, FaceLandmarksResult, HandLandmarksResult, PoseLandmarksResult
)

# Import from the new consolidated utilities
from ...utils.math import scale_value
from ...utils.general import AlwaysEqualProxy

logger = logging.getLogger(__name__)

MAX_POSITION_HISTORY = 50 # Matches MAX_LANDMARK_HISTORY for consistency
FLOAT_EQUALITY_TOLERANCE = 1e-6 # Same tolerance as base_delta_nodes
#TODO refactor to use CoordinateSystem
class LandmarkPositionBase:
    #NOTE: use x,y,and z for the landmark position rather than LandmarkPoint object for max flexibility
    """
    Base class for extracting position lists from specific landmark types.
    Outputs values corresponding to the input batch size, using defaults for missing items.
    Includes an 'is_valid' flag indicating successful extraction for each batch item.
    Processes only the detected object at the specified result_index per batch item.
    RETURN_TYPES declare the semantic type (FLOAT/BOOLEAN); the actual return value may be a list.
    """
    FUNCTION = "extract_positions"
    # Use base types for semantic declaration
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("x", "y", "z", "visibility", "presence", "is_valid") # Singular names
    CATEGORY = "Realtime Nodes/MediaPipe Vision/Common/Utils" 

    DEFAULT_VALUES = (0.0, 0.0, 0.0, 0.0, 0.0)
    LANDMARKS_TYPE = "ANY_LANDMARKS" # Placeholder

    # INPUT_TYPES are defined in subclasses

    def extract_positions(self, landmarks: list, landmark_index: int, result_index: int = 0, use_world_coordinates: bool = False) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[bool]]:
        """Extracts position and properties for the specified landmark across the entire batch,
           padding with defaults and providing validity flags for missing items."""

        x_coords = []
        y_coords = []
        z_coords = []
        vis_coords = []
        pres_coords = []
        is_valid_list = [] # New list to track validity

        if not isinstance(landmarks, list):
            raise TypeError(f"{self.__class__.__name__}: Input landmarks is not a list. Received type: {type(landmarks)}.")

        batch_size = len(landmarks)

        # Iterate through the batch (List[...])
        for batch_idx in range(batch_size):
            item_list = landmarks[batch_idx]
            
            current_x, current_y, current_z, current_vis, current_pres = self.DEFAULT_VALUES
            is_valid = False # Assume invalid until proven otherwise
            landmark_point: Optional[LandmarkPoint] = None
            target_landmarks_list: Optional[List[LandmarkPoint]] = None
            
            # Check if item_list is valid and contains at least one result item
            if isinstance(item_list, list) and item_list:
                # Get the result at the specified index if it exists
                if result_index < len(item_list):
                    result_item = item_list[result_index]

                    if result_item is not None:
                        try:
                            # --- Determine Structure and Extract Landmark List --- 
                            # Case 1: result_item is the list itself (FACE_LANDMARKS)
                            if isinstance(result_item, list):
                                target_landmarks_list = result_item
                                if use_world_coordinates and self.LANDMARKS_TYPE == "FACE_LANDMARKS":
                                    logger.warning(f"World coordinates requested but not available for FACE_LANDMARKS. Using image coordinates.")

                            # Case 2: result_item is an object with .landmarks (HAND/POSE_LANDMARKS)
                            elif hasattr(result_item, 'landmarks'):
                                landmarks_attr = getattr(result_item, 'landmarks', None)
                                if landmarks_attr is not None:
                                    use_world = (use_world_coordinates and
                                                 self.LANDMARKS_TYPE != "FACE_LANDMARKS" and
                                                 hasattr(result_item, 'world_landmarks') and
                                                 getattr(result_item, 'world_landmarks', None))
                                    if use_world:
                                        target_landmarks_list = result_item.world_landmarks
                                    else:
                                        target_landmarks_list = landmarks_attr
                            # else: target_landmarks_list remains None

                            # --- Find Landmark Point by Index ---
                            if isinstance(target_landmarks_list, list) and target_landmarks_list:
                                landmark_point = next((lm for lm in target_landmarks_list if lm.index == landmark_index), None)
                            
                        except Exception as e:
                             logger.error(f"{self.__class__.__name__}: Unexpected error processing batch item {batch_idx}: {e}", exc_info=True)
                             # Keep landmark_point as None, is_valid as False
                else:
                    logger.debug(f"{self.__class__.__name__}: Result index {result_index} exceeds available detections for batch item {batch_idx} (has {len(item_list)})")
            # else: item_list was not a list or was empty, keep defaults/invalid

            # --- Extract Values or Use Defaults, Set Validity ---
            if landmark_point:
                is_valid = True # Mark as valid since we found the point
                current_x = landmark_point.x
                current_y = landmark_point.y
                current_z = landmark_point.z
                current_vis = getattr(landmark_point, 'visibility', None)
                current_pres = getattr(landmark_point, 'presence', None)
                current_vis = current_vis if current_vis is not None else 0.0
                current_pres = current_pres if current_pres is not None else 0.0
            # else: defaults are used, is_valid remains False

            # Append values for this batch item
            x_coords.append(current_x)
            y_coords.append(current_y)
            z_coords.append(current_z)
            vis_coords.append(current_vis)
            pres_coords.append(current_pres)
            is_valid_list.append(is_valid)

        # Check if each list has only one element, if so return single values instead of lists
        if len(x_coords) == 1:
            return (x_coords[0], y_coords[0], z_coords[0], vis_coords[0], pres_coords[0], is_valid_list[0])
        
        return (x_coords, y_coords, z_coords, vis_coords, pres_coords, is_valid_list)
