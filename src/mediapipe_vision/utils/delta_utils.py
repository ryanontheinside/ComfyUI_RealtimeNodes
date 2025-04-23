import math
from typing import Optional

from ..types import LandmarkPoint # Assuming LandmarkPoint is in src/types.py

def calculate_euclidean_delta(p1: Optional[LandmarkPoint], p2: Optional[LandmarkPoint]) -> Optional[float]:
    """Calculates the Euclidean distance between two LandmarkPoint objects."""
    if p1 is None or p2 is None:
        return None # Cannot calculate delta if one point is missing
    
    delta_x = p1.x - p2.x
    delta_y = p1.y - p2.y
    delta_z = p1.z - p2.z
    
    distance = math.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
    return distance

def scale_value(value: Optional[float], 
                  delta_min: float, delta_max: float, 
                  output_min: float, output_max: float, 
                  clamp: bool = True) -> Optional[float]:
    """Linearly scales a value from one range to another."""
    if value is None:
        return None
    if delta_max == delta_min: # Avoid division by zero
        return output_min if value <= delta_min else output_max
        
    # Scale
    scaled = output_min + (value - delta_min) * (output_max - output_min) / (delta_max - delta_min)
    
    # Clamp
    if clamp:
        scaled = max(output_min, min(output_max, scaled))
        
    return scaled 