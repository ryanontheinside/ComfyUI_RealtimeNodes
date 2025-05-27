"""
Utility functions for coordinate processing
"""

import logging
from typing import List, Union, Tuple, Any, Dict

logger = logging.getLogger(__name__)


def validate_coordinate_input_types(x_in, y_in, z_in, class_name: str):
    is_list = isinstance(x_in, list)
    if is_list != isinstance(y_in, list) or is_list != isinstance(z_in, list):
        raise ValueError(f"{class_name}: Inputs x, y, z must be all floats or all lists.")
    return is_list


def cleanup_coordinate_histories(histories: Dict, current_batch_size: int, class_name: str):
    keys_to_remove = [k for k in histories if k >= current_batch_size]
    if keys_to_remove:
        # logger.debug(f"{class_name}: Removing histories for indices {keys_to_remove}")
        for k in keys_to_remove:
            del histories[k]


def cleanup_single_input_histories(histories: Dict, class_name: str):
    keys_to_remove = [k for k in histories if k != 0]
    if keys_to_remove:
        # logger.debug(f"{class_name}: Switching to single input, removing histories for indices {keys_to_remove}")
        for k in keys_to_remove:
            del histories[k]


def validate_coordinate_types_in_list(coords_list: List, index: int, class_name: str) -> bool:
    xi, yi, zi = coords_list[0][index], coords_list[1][index], coords_list[2][index]
    if not all(isinstance(c, (float, int)) for c in [xi, yi, zi]):
        logger.warning(f"{class_name}: Invalid coordinate type at index {index}. Using fallback value.")
        return False
    return True


def validate_single_coordinate_types(x_in, y_in, z_in, class_name: str) -> bool:
    if not all(isinstance(c, (float, int)) for c in [x_in, y_in, z_in]):
        logger.error(f"{class_name}: Invalid coordinate types for single float input.")
        return False
    return True


def check_required_arguments(args_dict: Dict, class_name: str) -> bool:
    if None in args_dict.values():
        logger.error(f"{class_name}: Missing required arguments.")
        return False
    return True


def handle_empty_coordinate_lists(current_batch_size: int, class_name: str) -> bool:
    if current_batch_size == 0:
        logger.warning(f"{class_name}: Received empty lists for coordinates.")
        return True
    return False


def standardize_inputs_to_lists(*inputs):
    return [inp if isinstance(inp, list) else [inp] for inp in inputs]


def parse_label_values(label_values: str) -> List[str]:
    values_list = []
    if label_values and label_values.strip():
        values_list = [v.strip() for v in label_values.split(',')]
    return values_list


def determine_batch_label(values_list: List[str], batch_index: int, label_prefix: str, batch_value_mode: str) -> str:
    if values_list:
        if batch_value_mode == "Index-based" and batch_index < len(values_list):
            # Use batch index to select value
            current_value = values_list[batch_index]
        else:
            # Default to first value
            current_value = values_list[0]
        current_label = f"{label_prefix}: {current_value}"
    else:
        current_label = label_prefix
    return current_label


def get_coordinate_space_and_dimensions(is_normalized: bool, image):
    from . import CoordinateSystem
    space = CoordinateSystem.NORMALIZED if is_normalized else CoordinateSystem.PIXEL
    dimensions = CoordinateSystem.get_dimensions_from_tensor(image)
    return space, dimensions


def validate_proximity_inputs(inputs: List, class_name: str) -> Tuple[bool, int]:
    is_list_input = isinstance(inputs[0], list)
    fallback_value = 0.0

    if any(isinstance(inp, list) != is_list_input for inp in inputs):
        raise ValueError("All coordinate inputs (x1..z2) must be of the same type (all floats or all lists).")

    if is_list_input:
        list_len = len(inputs[0])
        if any(len(inp) != list_len for inp in inputs if isinstance(inp, list)):
            raise ValueError("If inputs are lists, all coordinate lists (x1..z2) must have the same length.")
        if list_len == 0:
            logger.warning("Input coordinate lists are empty. Returning empty distance list.")
            return True, 0  # Return empty list indicator
        return False, list_len
    return False, 1


def validate_mask_coordinate_inputs(x, y, class_name: str) -> Tuple[bool, int]:
    is_x_list = isinstance(x, list)
    is_y_list = isinstance(y, list)

    if is_x_list != is_y_list:
        raise ValueError("Inputs x and y must be both floats or both lists.")

    if is_x_list:
        x_list = x
        y_list = y
        if len(x_list) != len(y_list):
            raise ValueError(f"x_list (len {len(x_list)}) and y_list (len {len(y_list)}) must have the same length.")
        if not x_list:  # Handle empty list
            logger.warning("Input coordinate lists are empty. Returning empty mask batch.")
            return True, 0  # Empty list indicator
        return True, len(x_list)
    return False, 1


def validate_individual_mask_coordinates(xi, yi, index: int, class_name: str) -> bool:
    if not isinstance(xi, (float, int)) or not isinstance(yi, (float, int)):
        logger.warning(
            f"Invalid coordinate type at index {index} (x={type(xi)}, y={type(yi)}). Skipping mask generation for this index."
        )
        return False
    return True 