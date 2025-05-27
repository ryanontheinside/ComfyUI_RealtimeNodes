import logging
import math
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch

from ...src.coordinates.coordinate_delta import BaseCoordinateDelta, MAX_POSITION_HISTORY, FLOAT_EQUALITY_TOLERANCE
from ...src.coordinates import coordinate_utils

# TODO: consider moving base nodes to src
# Import from the new consolidated utilities
from ...src.utils.math import scale_value

logger = logging.getLogger(__name__)


# --- Position Delta Control Nodes (Int/Float) (Handling Lists) ---
class CoordinateDeltaControlNode(BaseCoordinateDelta):
    """Base class for position-based delta control nodes (Int/Float).
    Handles single floats or lists of coordinates.
    Outputs FLOAT type (actual value is float or list[float]).
    """

    # Output semantic type is FLOAT (or INT for subclass)
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)  # Singular name
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
                "window_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": MAX_POSITION_HISTORY - 1,
                        "tooltip": "Number of frames back for delta",
                    },
                ),
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
        x_in, y_in, z_in = kwargs.get("x"), kwargs.get("y"), kwargs.get("z")
        window_size = kwargs.get("window_size", 1)
        delta_min = kwargs.get("delta_min")
        delta_max = kwargs.get("delta_max")
        output_min = kwargs.get(output_min_key)
        output_max = kwargs.get(output_max_key)
        clamp = kwargs.get("clamp")
        fallback_value = int(output_min) if self.IS_INT_OUTPUT else float(output_min)

        # Check required arguments
        required_args = {
            "x": x_in, "y": y_in, "z": z_in, "delta_min": delta_min, 
            "delta_max": delta_max, "output_min": output_min, "output_max": output_max, "clamp": clamp
        }
        if not coordinate_utils.check_required_arguments(required_args, self.__class__.__name__):
            return (fallback_value,)

        # Validate input types
        is_list = coordinate_utils.validate_coordinate_input_types(x_in, y_in, z_in, self.__class__.__name__)

        if is_list:
            current_batch_size = len(x_in)
            # Clean up histories
            coordinate_utils.cleanup_coordinate_histories(self._histories, current_batch_size, self.__class__.__name__)

            # Handle empty lists
            if coordinate_utils.handle_empty_coordinate_lists(current_batch_size, self.__class__.__name__):
                return ([],)

            # Process List Input
            results = []
            for i in range(current_batch_size):
                xi, yi, zi = x_in[i], y_in[i], z_in[i]
                if not coordinate_utils.validate_coordinate_types_in_list([[xi], [yi], [zi]], 0, self.__class__.__name__):
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
            # Clean up single input histories
            coordinate_utils.cleanup_single_input_histories(self._histories, self.__class__.__name__)

            # Validate single coordinate types
            if not coordinate_utils.validate_single_coordinate_types(x_in, y_in, z_in, self.__class__.__name__):
                return (fallback_value,)

            raw_delta = self._get_position_delta(float(x_in), float(y_in), float(z_in), window_size, item_index=0)
            scaled_value = scale_value(raw_delta, delta_min, delta_max, output_min, output_max, clamp)

            if self.IS_INT_OUTPUT:
                output_val = int(round(scaled_value)) if scaled_value is not None else int(output_min)
            else:
                output_val = scaled_value if scaled_value is not None else float(output_min)
            return (output_val,)


class CoordinateDeltaIntControlNode(CoordinateDeltaControlNode):
    RETURN_TYPES = ("INT",)  # Semantic type is INT
    RETURN_NAMES = ("value",)
    IS_INT_OUTPUT = True
    DEFAULT_OUTPUT_MIN = 0
    DEFAULT_OUTPUT_MAX = 100


class CoordinateDeltaFloatControlNode(CoordinateDeltaControlNode):
    RETURN_TYPES = ("FLOAT",)  # Semantic type is FLOAT
    RETURN_NAMES = ("value",)
    IS_INT_OUTPUT = False
    DEFAULT_OUTPUT_MIN = 0.0
    DEFAULT_OUTPUT_MAX = 1.0


# --- Position Delta Trigger Node (Handling Lists) ---
class CoordinateDeltaTriggerNode(BaseCoordinateDelta):
    """Outputs a BOOLEAN trigger if position delta crosses a threshold.
    Handles single floats or lists of coordinates.
    Outputs BOOLEAN type (actual value is boolean or list[boolean]).
    """

    RETURN_TYPES = ("BOOLEAN",)  # Semantic type
    RETURN_NAMES = ("trigger",)  # Singular name
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
        x_in, y_in, z_in = kwargs.get("x"), kwargs.get("y"), kwargs.get("z")
        window_size = kwargs.get("window_size", 1)
        threshold = kwargs.get("threshold")
        condition = kwargs.get("condition")
        fallback_value = False

        # Check required arguments
        required_args = {
            "x": x_in, "y": y_in, "z": z_in, "threshold": threshold, "condition": condition
        }
        if not coordinate_utils.check_required_arguments(required_args, self.__class__.__name__):
            return (fallback_value,)

        # Validate input types
        is_list = coordinate_utils.validate_coordinate_input_types(x_in, y_in, z_in, self.__class__.__name__)

        if is_list:
            current_batch_size = len(x_in)
            # Clean up histories
            coordinate_utils.cleanup_coordinate_histories(self._histories, current_batch_size, self.__class__.__name__)

            # Handle empty lists
            if coordinate_utils.handle_empty_coordinate_lists(current_batch_size, self.__class__.__name__):
                return ([],)

            # Process List Input
            results = []
            for i in range(current_batch_size):
                xi, yi, zi = x_in[i], y_in[i], z_in[i]
                if not coordinate_utils.validate_coordinate_types_in_list([[xi], [yi], [zi]], 0, self.__class__.__name__):
                    results.append(fallback_value)
                    continue

                raw_delta = self._get_position_delta(float(xi), float(yi), float(zi), window_size, item_index=i)
                triggered = fallback_value
                if raw_delta is not None:
                    if condition == "Above":
                        triggered = raw_delta > threshold
                    elif condition == "Below":
                        triggered = raw_delta < threshold
                    elif condition == "Equals":
                        triggered = abs(raw_delta - threshold) < FLOAT_EQUALITY_TOLERANCE
                    elif condition == "Not Equals":
                        triggered = abs(raw_delta - threshold) >= FLOAT_EQUALITY_TOLERANCE
                results.append(triggered)
            return (results,)

        else:
            # Clean up single input histories
            coordinate_utils.cleanup_single_input_histories(self._histories, self.__class__.__name__)

            # Validate single coordinate types
            if not coordinate_utils.validate_single_coordinate_types(x_in, y_in, z_in, self.__class__.__name__):
                return (fallback_value,)

            raw_delta = self._get_position_delta(float(x_in), float(y_in), float(z_in), window_size, item_index=0)
            triggered = fallback_value
            if raw_delta is not None:
                if condition == "Above":
                    triggered = raw_delta > threshold
                elif condition == "Below":
                    triggered = raw_delta < threshold
                elif condition == "Equals":
                    triggered = abs(raw_delta - threshold) < FLOAT_EQUALITY_TOLERANCE
                elif condition == "Not Equals":
                    triggered = abs(raw_delta - threshold) >= FLOAT_EQUALITY_TOLERANCE
            return (triggered,)


# --- Landmark Proximity Node (Handles Float or List Inputs) ---
class CoordinateProximityNode:
    """
    Calculates the Euclidean distance between landmark positions.
    Handles single coordinate sets or lists thereof.
    Outputs FLOAT type (actual value is float or list[float]).
    """

    CATEGORY = "Realtime Nodes/MediaPipe Vision/Common/Utils"
    FUNCTION = "calculate_distances"
    # Output semantic type is FLOAT
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("distance",)  # Singular name

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

    def calculate_distances(
        self,
        x1: Union[float, List[float]],
        y1: Union[float, List[float]],
        z1: Union[float, List[float]],
        x2: Union[float, List[float]],
        y2: Union[float, List[float]],
        z2: Union[float, List[float]],
    ) -> Union[float, List[float]]:  # Return type hint updated
        # Validate inputs
        inputs = [x1, y1, z1, x2, y2, z2]
        is_empty, list_len = coordinate_utils.validate_proximity_inputs(inputs, self.__class__.__name__)
        fallback_value = 0.0
        
        if is_empty:
            return ([],)  # Return empty list for FLOAT type

        is_list_input = isinstance(x1, list)

        if is_list_input:
            # Process List Input
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

                    dist = math.sqrt((_x2 - _x1) ** 2 + (_y2 - _y1) ** 2 + (_z2 - _z1) ** 2)
                    distances.append(dist)
                except Exception as e:
                    logger.error(f"Error calculating distance at index {i}: {e}", exc_info=True)
                    distances.append(fallback_value)
            return (distances,)  # Return list of results

        else:
            # Process Single Float Input
            _x1, _y1, _z1 = x1, y1, z1
            _x2, _y2, _z2 = x2, y2, z2
            if not all(isinstance(c, (float, int)) for c in [_x1, _y1, _z1, _x2, _y2, _z2]):
                logger.error(f"Invalid coordinate types for single float input.")
                return (fallback_value,)  # Return single fallback
            try:
                dist = math.sqrt((_x2 - _x1) ** 2 + (_y2 - _y1) ** 2 + (_z2 - _z1) ** 2)
                return (dist,)  # Return single float value
            except Exception as e:
                logger.error(f"Error calculating distance for single input: {e}", exc_info=True)
                return (fallback_value,)  # Return single fallback


class MaskFromCoordinate:
    """
    Creates geometric masks (circle or square) centered at landmark coordinates.
    Handles single (x, y) floats or lists (x_list, y_list) as input.
    Outputs a batch of masks (N, 1, H, W).
    """

    CATEGORY = "Realtime Nodes/MediaPipe Vision/Common/Masking"
    FUNCTION = "create_masks"  # Keep plural name for clarity
    RETURN_TYPES = ("MASK",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "forceInput": True,
                        "tooltip": "Normalized X coordinate (0.0-1.0) or List thereof",
                    },
                ),
                "y": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "forceInput": True,
                        "tooltip": "Normalized Y coordinate (0.0-1.0) or List thereof",
                    },
                ),
                "image_for_dimensions": ("IMAGE", {"tooltip": "Reference image to determine the output mask size"}),
                "radius": (
                    "INT",
                    {"default": 25, "min": 1, "max": 1024, "tooltip": "Radius of the mask shape in pixels"},
                ),
                "shape": (["Circle", "Square"], {"default": "Circle"}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    def create_masks(
        self,
        x: Union[float, List[float]],
        y: Union[float, List[float]],
        image_for_dimensions: torch.Tensor,
        radius: int,
        shape: str,
        invert: bool,
    ) -> Tuple[torch.Tensor]:
        # Validate coordinate inputs
        is_empty, num_coords = coordinate_utils.validate_mask_coordinate_inputs(x, y, self.__class__.__name__)
        
        if is_empty:
            # Determine shape for empty tensor
            if image_for_dimensions.dim() != 4:
                logger.error("Input image must be BHWC format to determine empty mask size.")
                return (torch.zeros((0, 1, 64, 64), dtype=torch.float32),)
            _, height, width, _ = image_for_dimensions.shape
            return (torch.zeros((0, 1, height, width), dtype=torch.float32, device=image_for_dimensions.device),)

        is_x_list = isinstance(x, list)
        if is_x_list:
            # Input is List[float]
            x_list = x
            y_list = y
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
            if not coordinate_utils.validate_individual_mask_coordinates(xi, yi, i, self.__class__.__name__):
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
                    cv2.circle(mask_np, center, radius, 1, -1)  # Default circle

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
