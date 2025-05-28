"""
Drawing utility functions
"""

import logging
from typing import List, Union, Tuple
import torch

logger = logging.getLogger(__name__)


def setup_batch_drawing(image: torch.Tensor):
    return image.shape[0], image.clone()


def handle_batch_mapping_coordinates(
    batch_index: int,
    batch_mapping: str,
    coord_lists: List[List],
    batch_size: int
) -> Tuple[bool, List]:
    if batch_mapping == "all-on-first" and batch_index > 0:
        return True, []
    
    if batch_mapping == "one-to-one":
        x_list, y_list = coord_lists[0], coord_lists[1]
        if batch_index >= len(x_list):
            return True, []
        
        # Build coordinates list efficiently
        batch_coords = [x_list[batch_index], y_list[batch_index]]
        if len(coord_lists) > 2:
            batch_coords.extend(coord_list[batch_index] for coord_list in coord_lists[2:])
        return False, batch_coords
    
    # Broadcast case
    return False, coord_lists




def setup_drawing_engine_and_coordinates(image: torch.Tensor, is_normalized: bool):
    from . import CoordinateSystem
    from .drawing_engine import DrawingEngine
    
    space = CoordinateSystem.NORMALIZED if is_normalized else CoordinateSystem.PIXEL
    dimensions = CoordinateSystem.get_dimensions_from_tensor(image)
    
    return DrawingEngine(), space, dimensions 