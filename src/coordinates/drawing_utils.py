"""
Drawing utility functions
"""

import logging
from typing import List, Union, Tuple
import torch

logger = logging.getLogger(__name__)


def setup_batch_drawing(image: torch.Tensor):
    batch_size = image.shape[0]
    output_batch = image.clone()
    return batch_size, output_batch


def handle_batch_mapping_coordinates(
    batch_index: int,
    batch_mapping: str,
    coord_lists: List[List],
    batch_size: int
) -> Tuple[bool, List]:
    x_list, y_list = coord_lists[0], coord_lists[1]
    
    if batch_mapping == "one-to-one":
        # One-to-one: each batch item gets one coordinate
        if batch_index < len(x_list):
            batch_coords = [x_list[batch_index], y_list[batch_index]]
            if len(coord_lists) > 2:  # Handle additional coordinates (x2, y2 for lines)
                batch_coords.extend([coord_list[batch_index] for coord_list in coord_lists[2:]])
            return False, batch_coords
        else:
            return True, []  # Skip if not enough coordinates
    elif batch_mapping == "all-on-first" and batch_index > 0:
        return True, []  # Skip all but the first item
    else:
        # Broadcast: all coordinates on all batches
        return False, coord_lists


def process_batch_item_drawing(
    output_batch: torch.Tensor,
    batch_index: int,
    drawing_function,
    drawing_args: dict
) -> torch.Tensor:
    # Create a single-item tensor for this batch
    single_image = output_batch[batch_index:batch_index+1]
    
    # Process this batch item
    result = drawing_function(image=single_image, **drawing_args)
    
    # Update the output batch
    output_batch[batch_index:batch_index+1] = result
    
    return output_batch


def setup_drawing_engine_and_coordinates(image: torch.Tensor, is_normalized: bool):
    from . import CoordinateSystem
    from .drawing_engine import DrawingEngine
    
    space = CoordinateSystem.NORMALIZED if is_normalized else CoordinateSystem.PIXEL
    dimensions = CoordinateSystem.get_dimensions_from_tensor(image)
    drawing_engine = DrawingEngine()
    
    return drawing_engine, space, dimensions 