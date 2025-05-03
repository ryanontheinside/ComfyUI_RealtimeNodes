"""
Transform utilities for manipulating affine transformation matrices.

Contains functions for creating and composing transformation matrices.
"""

import torch
import math


def create_identity_matrix(device="cpu"):
    """
    Create a 3x3 identity transformation matrix.

    Args:
        device: The torch device to place the matrix on

    Returns:
        3x3 identity matrix as torch.Tensor
    """
    return torch.eye(3, device=device)


def create_translation_matrix(dx, dy, device="cpu"):
    """
    Create a 3x3 translation matrix.

    Args:
        dx: Translation in x direction
        dy: Translation in y direction
        device: The torch device to place the matrix on

    Returns:
        3x3 translation matrix as torch.Tensor
    """
    matrix = torch.eye(3, device=device)
    matrix[0, 2] = dx
    matrix[1, 2] = dy
    return matrix


def create_rotation_matrix(angle_degrees, center_x=0.5, center_y=0.5, device="cpu"):
    """
    Create a 3x3 rotation matrix that rotates around the specified center point.

    Args:
        angle_degrees: Rotation angle in degrees
        center_x: X coordinate of rotation center (0-1 range)
        center_y: Y coordinate of rotation center (0-1 range)
        device: The torch device to place the matrix on

    Returns:
        3x3 rotation matrix as torch.Tensor
    """
    angle_rad = math.radians(angle_degrees)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    
    # Convert center from 0-1 range to -1 to 1 range
    center_x = (center_x * 2) - 1
    center_y = (center_y * 2) - 1
    
    # 1. Translate center to origin
    t1 = create_translation_matrix(-center_x, -center_y, device)
    
    # 2. Rotate around origin
    r = torch.eye(3, device=device)
    r[0, 0] = cos_theta
    r[0, 1] = -sin_theta
    r[1, 0] = sin_theta
    r[1, 1] = cos_theta
    
    # 3. Translate back
    t2 = create_translation_matrix(center_x, center_y, device)
    
    # Compose transformations: t2 @ r @ t1
    return torch.matmul(t2, torch.matmul(r, t1))


def create_scale_matrix(scale_x, scale_y, center_x=0.5, center_y=0.5, device="cpu"):
    """
    Create a 3x3 scaling matrix that scales around the specified center point.

    Args:
        scale_x: Scale factor in x direction
        scale_y: Scale factor in y direction
        center_x: X coordinate of scaling center (0-1 range)
        center_y: Y coordinate of scaling center (0-1 range)
        device: The torch device to place the matrix on

    Returns:
        3x3 scaling matrix as torch.Tensor
    """
    # Convert center from 0-1 range to -1 to 1 range
    center_x = (center_x * 2) - 1
    center_y = (center_y * 2) - 1
    
    # 1. Translate center to origin
    t1 = create_translation_matrix(-center_x, -center_y, device)
    
    # 2. Scale around origin
    s = torch.eye(3, device=device)
    s[0, 0] = scale_x
    s[1, 1] = scale_y
    
    # 3. Translate back
    t2 = create_translation_matrix(center_x, center_y, device)
    
    # Compose transformations: t2 @ s @ t1
    return torch.matmul(t2, torch.matmul(s, t1))


def compose_transforms(transform_a, transform_b):
    """
    Compose two transformation matrices. The result is equivalent to applying
    transform_b first, then transform_a.

    Args:
        transform_a: First transformation matrix
        transform_b: Second transformation matrix

    Returns:
        Composed transformation matrix
    """
    return torch.matmul(transform_a, transform_b)


def normalize_homogeneous_coordinates(coords):
    """
    Normalize homogeneous coordinates by dividing by the last component.

    Args:
        coords: Tensor of homogeneous coordinates [..., 3]

    Returns:
        Normalized coordinates
    """
    # Extract last component
    w = coords[..., 2:3]
    
    # Handle zeros (avoid division by zero)
    mask = w != 0
    
    # Create a copy of the coordinates
    normalized = coords.clone()
    
    # Apply normalization only to non-zero denominator elements
    # We need to expand the mask to match the coords dimension
    for i in range(coords.shape[-1]):
        normalized[..., i] = torch.where(mask.squeeze(-1), coords[..., i] / w.squeeze(-1), coords[..., i])
    
    return normalized


def apply_transform(points, transform_matrix):
    """
    Apply a transformation matrix to a set of points.

    Args:
        points: Tensor of points [..., 2]
        transform_matrix: 3x3 transformation matrix

    Returns:
        Transformed points
    """
    # Convert to homogeneous coordinates
    homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    
    # Apply transformation
    transformed = torch.matmul(homogeneous, transform_matrix.T)
    
    # Normalize and return x, y coordinates
    normalized = normalize_homogeneous_coordinates(transformed)
    return normalized[..., :2] 