"""
Unified utilities package for ComfyUI RealTimeNodes.
"""

# General utilities
from .general import AlwaysEqualProxy

# Image utilities
from .image import convert_to_cv2, convert_to_tensor, create_mask_from_points

# Math utilities
from .math import (
    calculate_euclidean_distance,
    calculate_euclidean_distance_2d,
    scale_value,
)

# Timing utilities
from .timing import TimestampProvider

# For backward compatibility, expose the original names
# This allows existing code to continue working while we transition
image_to_cv2 = convert_to_cv2
cv2_to_image = convert_to_tensor
