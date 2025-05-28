"""
Utility functions for coordinate processing
"""

import logging
from typing import List, Union, Tuple, Any, Dict

logger = logging.getLogger(__name__)


def validate_coordinate_input_types(x_in, y_in, z_in, class_name: str):
    x_is_list = isinstance(x_in, list)
    y_is_list = isinstance(y_in, list)
    z_is_list = isinstance(z_in, list)
    if x_is_list != y_is_list or x_is_list != z_is_list:
        raise ValueError(f"{class_name}: Inputs x, y, z must be all floats or all lists.")
    return x_is_list


def cleanup_coordinate_histories(histories: Dict, current_batch_size: int, class_name: str):
    if not histories:
        return
    for k in list(histories.keys()):
        if k >= current_batch_size:
            del histories[k]


def cleanup_single_input_histories(histories: Dict, class_name: str):
    if len(histories) <= 1:
        return
    keep_zero = 0 in histories
    histories.clear()
    if keep_zero:
        pass  # Will be recreated as needed


def standardize_inputs_to_lists(*inputs):
    if all(isinstance(inp, list) for inp in inputs):
        return inputs
    return tuple(inp if isinstance(inp, list) else [inp] for inp in inputs)


def parse_label_values(label_values: str) -> List[str]:
    if not label_values or not label_values.strip():
        return []
    return [v.strip() for v in label_values.split(',') if v.strip()]


def determine_batch_label(values_list: List[str], batch_index: int, label_prefix: str, batch_value_mode: str) -> str:
    if not values_list:
        return label_prefix
    
    if batch_value_mode == "Index-based" and batch_index < len(values_list):
        return f"{label_prefix}: {values_list[batch_index]}"
    return f"{label_prefix}: {values_list[0]}"


def get_coordinate_space_and_dimensions(is_normalized: bool, image):
    from . import CoordinateSystem
    space = CoordinateSystem.NORMALIZED if is_normalized else CoordinateSystem.PIXEL
    dimensions = CoordinateSystem.get_dimensions_from_tensor(image)
    return space, dimensions


def validate_proximity_inputs(inputs: List, class_name: str) -> Tuple[bool, int]:
    if not inputs:
        return True, 0
    
    first_input = inputs[0]
    is_list_input = isinstance(first_input, list)

    for inp in inputs[1:]:
        if isinstance(inp, list) != is_list_input:
            raise ValueError("All coordinate inputs (x1..z2) must be of the same type (all floats or all lists).")

    if is_list_input:
        first_len = len(first_input)
        if first_len == 0:
            return True, 0
        
        for inp in inputs[1:]:
            if isinstance(inp, list) and len(inp) != first_len:
                raise ValueError("If inputs are lists, all coordinate lists (x1..z2) must have the same length.")
        
        return False, first_len
    return False, 1


def validate_mask_coordinate_inputs(x, y, class_name: str) -> Tuple[bool, int]:
    x_is_list = isinstance(x, list)
    
    if x_is_list != isinstance(y, list):
        raise ValueError("Inputs x and y must be both floats or both lists.")

    if x_is_list:
        x_len = len(x)
        if x_len != len(y):
            raise ValueError(f"x_list (len {x_len}) and y_list (len {len(y)}) must have the same length.")
        if x_len == 0:
            return True, 0
        return True, x_len
    return False, 1 