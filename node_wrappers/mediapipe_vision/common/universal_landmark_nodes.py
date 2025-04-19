import logging
from typing import Optional, List, Any, Tuple, Union
import math
import torch
import numpy as np
import cv2
import collections
import typing # Add typing for Dict hint

# Assuming types are correctly imported relative to this file's location
from ....src.mediapipe_vision.types import (
    FACE_LANDMARKS, HAND_LANDMARKS, POSE_LANDMARKS,
    LandmarkPoint, FaceLandmarksResult, HandLandmarksResult, PoseLandmarksResult
)

# Re-import needed utils if not available globally
from ....src.mediapipe_vision.utils.delta_utils import scale_value
from ....src.mediapipe_vision.utils.comfy_utils import AlwaysEqualProxy

logger = logging.getLogger(__name__)

MAX_POSITION_HISTORY = 50 # Matches MAX_LANDMARK_HISTORY for consistency
FLOAT_EQUALITY_TOLERANCE = 1e-6 # Same tolerance as base_delta_nodes

class LandmarkPositionBaseNode:
    #NOTE: use x,y,and z for the landmark position rather than LandmarkPoint object for max flexibility
    """
    Base class for extracting position lists from specific landmark types.
    Outputs values corresponding to the input batch size, using defaults for missing items.
    Includes an 'is_valid' flag indicating successful extraction for each batch item.
    Processes only the first detected object (e.g., face, hand) per batch item.
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

    def extract_positions(self, landmarks: list, landmark_index: int, use_world_coordinates: bool = False) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[bool]]:
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
                result_item = item_list[0] # Process only the first detected object

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

        return (x_coords, y_coords, z_coords, vis_coords, pres_coords, is_valid_list)


# --- Base Position Delta Logic (Handling Multiple Histories) ---
class BasePositionDeltaNode:
    """Base class for nodes calculating delta from position history.
       Handles single floats or lists of coordinates.
    """
    FUNCTION = "execute"

    def __init__(self):
        # Store history deques keyed by item index from input list
        # Key 0 is used for single float inputs
        self._histories: typing.Dict[int, collections.deque] = {}

    def _get_or_create_history(self, item_index: int) -> collections.deque:
        """Gets the deque for a specific index, creating it if needed."""
        if item_index not in self._histories:
            self._histories[item_index] = collections.deque(maxlen=MAX_POSITION_HISTORY)
        return self._histories[item_index]

    def _get_position_delta(self, x: float, y: float, z: float, window_size: int, item_index: int = 0) -> Optional[float]:
        """Calculates delta using the history deque for the given item_index."""
        history_deque = self._get_or_create_history(item_index)
        
        window_size = max(1, min(window_size, MAX_POSITION_HISTORY - 1))
        current_pos = (x, y, z)
        history_deque.append(current_pos)
        
        if len(history_deque) > window_size:
            pos_now = history_deque[-1]
            pos_past = history_deque[-(window_size + 1)] # Correct indexing
            if pos_now is None or pos_past is None:
                 logger.warning(f"{self.__class__.__name__}[{item_index}]: Missing position data in history window.")
                 return None
            delta = math.sqrt((pos_now[0] - pos_past[0])**2 +
                              (pos_now[1] - pos_past[1])**2 +
                              (pos_now[2] - pos_past[2])**2)
            return delta
        else:
            # logger.debug(f"{self.__class__.__name__}[{item_index}]: Not enough history ({len(history_deque)}/{window_size+1}).")
            return None # Not enough history yet
            
    # Remove _handle_input_coords, logic moved to execute

# --- Position Delta Control Nodes (Int/Float) (Handling Lists) ---
class PositionDeltaControlNode(BasePositionDeltaNode):
    """Base class for position-based delta control nodes (Int/Float).
       Handles single floats or lists of coordinates.
       Outputs FLOAT type (actual value is float or list[float]).
    """
    # Output semantic type is FLOAT (or INT for subclass)
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",) # Singular name
    IS_INT_OUTPUT = False
    DEFAULT_DELTA_MIN = 0.0
    DEFAULT_DELTA_MAX = 0.1 
    DEFAULT_OUTPUT_MIN = 0.0
    DEFAULT_OUTPUT_MAX = 1.0
    DEFAULT_CLAMP = True
    CATEGORY = "Realtime Nodes/MediaPipe Vision/Common/Delta"

    @classmethod
    def INPUT_TYPES(cls):
        # Input types remain the same
        output_type_str = "INT" if cls.IS_INT_OUTPUT else "FLOAT"
        output_min_key = f"output_min_{output_type_str.lower()}"
        output_max_key = f"output_max_{output_type_str.lower()}"
        return {
            "required": {
                "x": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "y": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "z": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "window_size": ("INT", { "default": 1, "min": 1, "max": MAX_POSITION_HISTORY - 1, "tooltip": "Number of frames back for delta"}),
                "delta_min": ("FLOAT", {"default": cls.DEFAULT_DELTA_MIN, "step": 0.01}),
                "delta_max": ("FLOAT", {"default": cls.DEFAULT_DELTA_MAX, "step": 0.01}),
                output_min_key: (output_type_str, {"default": cls.DEFAULT_OUTPUT_MIN}),
                output_max_key: (output_type_str, {"default": cls.DEFAULT_OUTPUT_MAX}),
                "clamp": ("BOOLEAN", {"default": cls.DEFAULT_CLAMP}),
            }
        }

    def execute(self, **kwargs):
        output_type_str = "INT" if self.IS_INT_OUTPUT else "FLOAT"
        output_min_key = f"output_min_{output_type_str.lower()}"
        output_max_key = f"output_max_{output_type_str.lower()}"
        x_in, y_in, z_in = kwargs.get('x'), kwargs.get('y'), kwargs.get('z')
        window_size = kwargs.get('window_size', 1)
        delta_min = kwargs.get('delta_min')
        delta_max = kwargs.get('delta_max')
        output_min = kwargs.get(output_min_key)
        output_max = kwargs.get(output_max_key)
        clamp = kwargs.get('clamp')
        fallback_value = int(output_min) if self.IS_INT_OUTPUT else float(output_min)

        if None in [x_in, y_in, z_in, delta_min, delta_max, output_min, output_max, clamp]:
            logger.error(f"{self.__class__.__name__}: Missing required arguments.")
            return (fallback_value,)

        # --- Input Type Handling & History Cleanup ---
        is_list = isinstance(x_in, list)
        if is_list != isinstance(y_in, list) or is_list != isinstance(z_in, list):
            raise ValueError(f"{self.__class__.__name__}: Inputs x, y, z must be all floats or all lists.")

        if is_list:
            current_batch_size = len(x_in)
            # Remove histories for indices no longer present
            keys_to_remove = [k for k in self._histories if k >= current_batch_size]
            if keys_to_remove:
                # logger.debug(f"{self.__class__.__name__}: Removing histories for indices {keys_to_remove}")
                for k in keys_to_remove:
                    del self._histories[k]
            
            if current_batch_size == 0:
                 logger.warning(f"{self.__class__.__name__}: Received empty lists for coordinates.")
                 return ([],)
            
            # Process List Input
            results = []
            for i in range(current_batch_size):
                xi, yi, zi = x_in[i], y_in[i], z_in[i]
                if not all(isinstance(c, (float, int)) for c in [xi, yi, zi]):
                    logger.warning(f"{self.__class__.__name__}: Invalid coordinate type at index {i}. Using fallback value.")
                    results.append(fallback_value)
                    continue
                
                raw_delta = self._get_position_delta(float(xi), float(yi), float(zi), window_size, item_index=i)
                scaled_value = scale_value(raw_delta, delta_min, delta_max, output_min, output_max, clamp)
                
                if self.IS_INT_OUTPUT:
                    output_val = int(round(scaled_value)) if scaled_value is not None else int(output_min)
                else:
                    output_val = scaled_value if scaled_value is not None else float(output_min)
                results.append(output_val)
            return (results,)
            
        else:
            # Single Float Input: Remove all histories except for index 0
            keys_to_remove = [k for k in self._histories if k != 0]
            if keys_to_remove:
                # logger.debug(f"{self.__class__.__name__}: Switching to single input, removing histories for indices {keys_to_remove}")
                for k in keys_to_remove:
                    del self._histories[k]
                    
            # Process Single Float Input
            if not all(isinstance(c, (float, int)) for c in [x_in, y_in, z_in]):
                 logger.error(f"{self.__class__.__name__}: Invalid coordinate types for single float input.")
                 return (fallback_value,)
                 
            raw_delta = self._get_position_delta(float(x_in), float(y_in), float(z_in), window_size, item_index=0)
            scaled_value = scale_value(raw_delta, delta_min, delta_max, output_min, output_max, clamp)
            
            if self.IS_INT_OUTPUT:
                output_val = int(round(scaled_value)) if scaled_value is not None else int(output_min)
            else:
                output_val = scaled_value if scaled_value is not None else float(output_min)
            return (output_val,)

class PositionDeltaIntControlNode(PositionDeltaControlNode):
    RETURN_TYPES = ("INT",) # Semantic type is INT
    RETURN_NAMES = ("value",)
    IS_INT_OUTPUT = True
    DEFAULT_OUTPUT_MIN = 0
    DEFAULT_OUTPUT_MAX = 100

class PositionDeltaFloatControlNode(PositionDeltaControlNode):
    RETURN_TYPES = ("FLOAT",) # Semantic type is FLOAT
    RETURN_NAMES = ("value",)
    IS_INT_OUTPUT = False
    DEFAULT_OUTPUT_MIN = 0.0
    DEFAULT_OUTPUT_MAX = 1.0

# --- Position Delta Trigger Node (Handling Lists) ---
class PositionDeltaTriggerNode(BasePositionDeltaNode):
    """Outputs a BOOLEAN trigger if position delta crosses a threshold.
       Handles single floats or lists of coordinates.
       Outputs BOOLEAN type (actual value is boolean or list[boolean]).
    """
    RETURN_TYPES = ("BOOLEAN",) # Semantic type
    RETURN_NAMES = ("trigger",) # Singular name
    DEFAULT_THRESHOLD = 0.05
    CATEGORY = "Realtime Nodes/MediaPipe Vision/Common/Delta"

    @classmethod
    def INPUT_TYPES(cls):
        # Inputs remain the same
        return {
            "required": {
                "x": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "y": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "z": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "window_size": ("INT", {"default": 1, "min": 1, "max": MAX_POSITION_HISTORY - 1}),
                "threshold": ("FLOAT", {"default": cls.DEFAULT_THRESHOLD, "step": 0.01}),
                "condition": (["Above", "Below", "Equals", "Not Equals"],),
            }
        }

    def execute(self, **kwargs):
        x_in, y_in, z_in = kwargs.get('x'), kwargs.get('y'), kwargs.get('z')
        window_size = kwargs.get('window_size', 1)
        threshold = kwargs.get('threshold')
        condition = kwargs.get('condition')
        fallback_value = False

        if None in [x_in, y_in, z_in, threshold, condition]:
             logger.error(f"{self.__class__.__name__}: Missing required arguments.")
             return (fallback_value,)

        # --- Input Type Handling & History Cleanup ---
        is_list = isinstance(x_in, list)
        if is_list != isinstance(y_in, list) or is_list != isinstance(z_in, list):
            raise ValueError(f"{self.__class__.__name__}: Inputs x, y, z must be all floats or all lists.")

        if is_list:
            current_batch_size = len(x_in)
            # Remove histories for indices no longer present
            keys_to_remove = [k for k in self._histories if k >= current_batch_size]
            if keys_to_remove:
                # logger.debug(f"{self.__class__.__name__}: Removing histories for indices {keys_to_remove}")
                for k in keys_to_remove:
                    del self._histories[k]
                    
            if current_batch_size == 0:
                 logger.warning(f"{self.__class__.__name__}: Received empty lists for coordinates.")
                 return ([],)
                 
            # Process List Input
            results = []
            for i in range(current_batch_size):
                xi, yi, zi = x_in[i], y_in[i], z_in[i]
                if not all(isinstance(c, (float, int)) for c in [xi, yi, zi]):
                    logger.warning(f"{self.__class__.__name__}: Invalid coordinate type at index {i}. Using fallback value.")
                    results.append(fallback_value)
                    continue
                    
                raw_delta = self._get_position_delta(float(xi), float(yi), float(zi), window_size, item_index=i)
                triggered = fallback_value
                if raw_delta is not None:
                    if condition == "Above": triggered = raw_delta > threshold
                    elif condition == "Below": triggered = raw_delta < threshold
                    elif condition == "Equals": triggered = abs(raw_delta - threshold) < FLOAT_EQUALITY_TOLERANCE
                    elif condition == "Not Equals": triggered = abs(raw_delta - threshold) >= FLOAT_EQUALITY_TOLERANCE
                results.append(triggered)
            return (results,)
            
        else:
             # Single Float Input: Remove all histories except for index 0
            keys_to_remove = [k for k in self._histories if k != 0]
            if keys_to_remove:
                # logger.debug(f"{self.__class__.__name__}: Switching to single input, removing histories for indices {keys_to_remove}")
                for k in keys_to_remove:
                    del self._histories[k]
                    
            # Process Single Float Input
            if not all(isinstance(c, (float, int)) for c in [x_in, y_in, z_in]):
                 logger.error(f"{self.__class__.__name__}: Invalid coordinate types for single float input.")
                 return (fallback_value,)
                 
            raw_delta = self._get_position_delta(float(x_in), float(y_in), float(z_in), window_size, item_index=0)
            triggered = fallback_value
            if raw_delta is not None:
                if condition == "Above": triggered = raw_delta > threshold
                elif condition == "Below": triggered = raw_delta < threshold
                elif condition == "Equals": triggered = abs(raw_delta - threshold) < FLOAT_EQUALITY_TOLERANCE
                elif condition == "Not Equals": triggered = abs(raw_delta - threshold) >= FLOAT_EQUALITY_TOLERANCE
            return (triggered,)

# --- Landmark Proximity Node (Handles Float or List Inputs) ---
class LandmarkProximityNode:
    """
    Calculates the Euclidean distance between landmark positions.
    Handles single coordinate sets or lists thereof.
    Outputs FLOAT type (actual value is float or list[float]).
    """
    CATEGORY = "Realtime Nodes/MediaPipe Vision/Common/Utils"
    FUNCTION = "calculate_distances"
    # Output semantic type is FLOAT
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("distance",) # Singular name

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Keep FLOAT type
                "x1": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "y1": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "z1": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "x2": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "y2": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "z2": ("FLOAT", {"default": 0.0, "forceInput": True}),
            }
        }

    def calculate_distances(self, x1: Union[float, List[float]], y1: Union[float, List[float]], z1: Union[float, List[float]],
                            x2: Union[float, List[float]], y2: Union[float, List[float]], z2: Union[float, List[float]]) -> Union[float, List[float]]: # Return type hint updated

        # --- Input Type Handling ---
        inputs = [x1, y1, z1, x2, y2, z2]
        is_list_input = isinstance(x1, list) 
        fallback_value = 0.0

        if any(isinstance(inp, list) != is_list_input for inp in inputs):
             raise ValueError("All coordinate inputs (x1..z2) must be of the same type (all floats or all lists).")

        if is_list_input:
            # Process List Input
            list_len = len(x1)
            if any(len(inp) != list_len for inp in inputs if isinstance(inp, list)):
                 raise ValueError("If inputs are lists, all coordinate lists (x1..z2) must have the same length.")
            if list_len == 0:
                logger.warning("Input coordinate lists are empty. Returning empty distance list.")
                return ([],) # Return empty list for FLOAT type
            num_calcs = list_len
            x1_list, y1_list, z1_list = x1, y1, z1
            x2_list, y2_list, z2_list = x2, y2, z2
            
            distances = []
            for i in range(num_calcs):
                try:
                    _x1, _y1, _z1 = x1_list[i], y1_list[i], z1_list[i]
                    _x2, _y2, _z2 = x2_list[i], y2_list[i], z2_list[i]
                    
                    if not all(isinstance(c, (float, int)) for c in [_x1, _y1, _z1, _x2, _y2, _z2]):
                         logger.warning(f"Invalid coordinate type found at index {i}. Using fallback distance {fallback_value}.")
                         distances.append(fallback_value)
                         continue
                         
                    dist = math.sqrt((_x2 - _x1)**2 + (_y2 - _y1)**2 + (_z2 - _z1)**2)
                    distances.append(dist)
                except Exception as e:
                    logger.error(f"Error calculating distance at index {i}: {e}", exc_info=True)
                    distances.append(fallback_value) 
            return (distances,) # Return list of results
            
        else:
            # Process Single Float Input
            _x1, _y1, _z1 = x1, y1, z1
            _x2, _y2, _z2 = x2, y2, z2
            if not all(isinstance(c, (float, int)) for c in [_x1, _y1, _z1, _x2, _y2, _z2]):
                 logger.error(f"Invalid coordinate types for single float input.")
                 return (fallback_value,) # Return single fallback
            try:
                dist = math.sqrt((_x2 - _x1)**2 + (_y2 - _y1)**2 + (_z2 - _z1)**2)
                return (dist,) # Return single float value
            except Exception as e:
                logger.error(f"Error calculating distance for single input: {e}", exc_info=True)
                return (fallback_value,) # Return single fallback

# --- Universal Landmark Mask Node (Handles Float or List Inputs) ---
class UniversalLandmarkMaskNode:
    """
    Creates geometric masks (circle or square) centered at landmark coordinates.
    Handles single (x, y) floats or lists (x_list, y_list) as input.
    Outputs a batch of masks (N, 1, H, W).
    """
    CATEGORY = "Realtime Nodes/MediaPipe Vision/Common/Masking"
    FUNCTION = "create_masks" # Keep plural name for clarity
    RETURN_TYPES = ("MASK",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("FLOAT", {"default": 0.5, "forceInput": True, "tooltip": "Normalized X coordinate (0.0-1.0) or List thereof"}),
                "y": ("FLOAT", {"default": 0.5, "forceInput": True, "tooltip": "Normalized Y coordinate (0.0-1.0) or List thereof"}),
                "image_for_dimensions": ("IMAGE", {"tooltip": "Reference image to determine the output mask size"}),
                "radius": ("INT", {"default": 25, "min": 1, "max": 1024, "tooltip": "Radius of the mask shape in pixels"}),
                "shape": (["Circle", "Square"], {"default": "Circle"}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    def create_masks(self, x: Union[float, List[float]], y: Union[float, List[float]], image_for_dimensions: torch.Tensor, radius: int, shape: str, invert: bool) -> Tuple[torch.Tensor]:

        # --- Input Type Handling ---
        is_x_list = isinstance(x, list)
        is_y_list = isinstance(y, list)

        if is_x_list != is_y_list:
            raise ValueError("Inputs x and y must be both floats or both lists.")

        if is_x_list:
            # Input is List[float]
            x_list = x
            y_list = y
            if len(x_list) != len(y_list):
                raise ValueError(f"x_list (len {len(x_list)}) and y_list (len {len(y_list)}) must have the same length.")
            if not x_list: # Handle empty list
                 logger.warning("Input coordinate lists are empty. Returning empty mask batch.")
                 # Determine shape for empty tensor
                 if image_for_dimensions.dim() != 4:
                      logger.error("Input image must be BHWC format to determine empty mask size.")
                      return (torch.zeros((0, 1, 64, 64), dtype=torch.float32),)
                 _, height, width, _ = image_for_dimensions.shape
                 return (torch.zeros((0, 1, height, width), dtype=torch.float32, device=image_for_dimensions.device),)
            num_coords = len(x_list)
        else:
            # Input is float - wrap in lists for unified processing
            x_list = [x]
            y_list = [y]
            num_coords = 1

        # --- Get Dimensions --- 
        if image_for_dimensions.dim() != 4:
            logger.error("Input image must be BHWC format.")
            # Return empty batch matching input size
            return (torch.zeros((num_coords, 1, 64, 64), dtype=torch.float32),)
        _, height, width, _ = image_for_dimensions.shape
        device = image_for_dimensions.device

        # --- Generate Masks --- 
        output_masks = []
        for i in range(num_coords):
            xi = x_list[i]
            yi = y_list[i]

            # Validate individual coordinates
            if not isinstance(xi, (float, int)) or not isinstance(yi, (float, int)):
                 logger.warning(f"Invalid coordinate type at index {i} (x={type(xi)}, y={type(yi)}). Skipping mask generation for this index.")
                 mask_tensor = torch.zeros((1, height, width), dtype=torch.float32, device=device)
                 output_masks.append(mask_tensor)
                 continue

            # Calculate pixel coordinates
            px = max(0, min(width - 1, int(xi * width)))
            py = max(0, min(height - 1, int(yi * height)))

            mask_np = np.zeros((height, width), dtype=np.uint8)
            try:
                center = (px, py)
                if shape == "Circle":
                    cv2.circle(mask_np, center, radius, 1, -1)
                elif shape == "Square":
                    half_size = radius
                    top_left = (max(0, px - half_size), max(0, py - half_size))
                    bottom_right = (min(width - 1, px + half_size), min(height - 1, py + half_size))
                    cv2.rectangle(mask_np, top_left, bottom_right, 1, -1)
                else:
                    cv2.circle(mask_np, center, radius, 1, -1) # Default circle

                if invert:
                    mask_np = 1 - mask_np

            except Exception as e:
                logger.error(f"Error drawing mask shape at index {i}: {e}", exc_info=True)
                # mask_np remains zeros

            # Convert to tensor (1, H, W)
            mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).to(device)
            output_masks.append(mask_tensor)

        # Stack into batch (N, 1, H, W)
        output_batch = torch.stack(output_masks, dim=0)
        return (output_batch,)

# --- Node Mappings (Revert Delta node display names) ---
NODE_CLASS_MAPPINGS = {
    "LandmarkProximity": LandmarkProximityNode,
    "UniversalLandmarkMask": UniversalLandmarkMaskNode,
    "PositionDeltaIntControl": PositionDeltaIntControlNode,
    "PositionDeltaFloatControl": PositionDeltaFloatControlNode,
    "PositionDeltaTrigger": PositionDeltaTriggerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LandmarkProximity": "Landmark Proximity Calculator", 
    "UniversalLandmarkMask": "Universal Landmark Mask Generator (Batch)",
    "PositionDeltaIntControl": "Position Delta Int Control", 
    "PositionDeltaFloatControl": "Position Delta Float Control", 
    "PositionDeltaTrigger": "Position Delta Trigger", 
}