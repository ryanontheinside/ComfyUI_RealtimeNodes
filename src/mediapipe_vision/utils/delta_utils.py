import math
from typing import Optional

from ...utils.math import scale_value
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

# scale_value is now imported from utils.math 