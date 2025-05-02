"""
Unified utilities package for ComfyUI RealTimeNodes.
"""

# General utilities
from .general import AlwaysEqualProxy

# Image utilities
from .image import convert_to_cv2, convert_to_tensor, create_mask_from_points, flow_to_rgb, gaussian_blur_2d

# Math utilities
from .math import (
    calculate_euclidean_distance,
    calculate_euclidean_distance_2d,
    scale_value,
)

# Timing utilities
from .timing import TimestampProvider

# Similar image filter
from .similar_image_filter import SimilarImageFilter

# Transform utilities
from .transforms import (
    create_identity_matrix,
    create_translation_matrix,
    create_rotation_matrix,
    create_scale_matrix,
    compose_transforms,
    normalize_homogeneous_coordinates,
    apply_transform
)

# Warp utilities
from .warp import (
    create_sampling_grid,
    transform_grid,
    warp_image,
    warp_latent
)

# Real-time flownets
from .realtime_flownets import RealTimeFlowNet 

# For backward compatibility, expose the original names
# This allows existing code to continue working while we transition
image_to_cv2 = convert_to_cv2
cv2_to_image = convert_to_tensor
