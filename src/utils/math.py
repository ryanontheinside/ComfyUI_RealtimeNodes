"""
Math utility functions for RealTimeNodes.

Contains functions for mathematical operations including:
- Value scaling and normalization
- Distance calculations
- Other general math operations
"""

import math
from typing import Optional, Tuple

def scale_value(value: Optional[float], 
                input_min: float, input_max: float, 
                output_min: float, output_max: float, 
                clamp: bool = True) -> Optional[float]:
    """
    Linearly scales a value from one range to another.
    
    Args:
        value: The value to scale
        input_min: Minimum value of input range
        input_max: Maximum value of input range
        output_min: Minimum value of output range
        output_max: Maximum value of output range
        clamp: Whether to clamp the output to the output range
        
    Returns:
        Scaled value, or None if input was None
    """
    if value is None:
        return None
    if input_max == input_min:  # Avoid division by zero
        return output_min if value <= input_min else output_max
        
    # Scale
    scaled = output_min + (value - input_min) * (output_max - output_min) / (input_max - input_min)
    
    # Clamp
    if clamp:
        scaled = max(output_min, min(output_max, scaled))
        
    return scaled

def calculate_euclidean_distance(point1: Tuple[float, float, float], 
                                point2: Tuple[float, float, float]) -> float:
    """
    Calculates the Euclidean distance between two 3D points.
    
    Args:
        point1: First point as (x, y, z)
        point2: Second point as (x, y, z)
        
    Returns:
        Euclidean distance between the points
    """
    delta_x = point1[0] - point2[0]
    delta_y = point1[1] - point2[1]
    delta_z = point1[2] - point2[2]
    
    distance = math.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
    return distance

def calculate_euclidean_distance_2d(point1: Tuple[float, float], 
                                   point2: Tuple[float, float]) -> float:
    """
    Calculates the Euclidean distance between two 2D points.
    
    Args:
        point1: First point as (x, y)
        point2: Second point as (x, y)
        
    Returns:
        Euclidean distance between the points
    """
    delta_x = point1[0] - point2[0]
    delta_y = point1[1] - point2[1]
    
    distance = math.sqrt(delta_x**2 + delta_y**2)
    return distance 