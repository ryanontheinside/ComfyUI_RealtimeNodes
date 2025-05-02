"""
Warping utilities for transforming images and latents using transformation matrices.

Contains functions for grid-based warping operations.
"""

import torch
import torch.nn.functional as F
from .transforms import normalize_homogeneous_coordinates


def create_sampling_grid(height, width, device="cpu"):
    """
    Create a normalized sampling grid for an image of the given dimensions.
    
    Args:
        height: Image height
        width: Image width
        device: The torch device to place the grid on
        
    Returns:
        Normalized grid of shape [height, width, 2] with values in range [-1, 1]
    """
    # Create coordinate grid in pixel space
    y_coords = torch.linspace(0, height - 1, height, device=device)
    x_coords = torch.linspace(0, width - 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    
    # Convert to normalized coordinates in range [-1, 1]
    grid_x = 2.0 * grid_x / (width - 1) - 1.0
    grid_y = 2.0 * grid_y / (height - 1) - 1.0
    
    # Stack into grid of shape [height, width, 2]
    grid = torch.stack([grid_x, grid_y], dim=-1)
    
    return grid


def transform_grid(grid, transform_matrix):
    """
    Apply a transformation matrix to a sampling grid.
    
    The transformation matrix is applied in reverse (inverse) since we're creating
    a sampling grid - we need to know where to sample from in the source image.
    
    Args:
        grid: Sampling grid of shape [..., 2]
        transform_matrix: 3x3 transformation matrix
        
    Returns:
        Transformed grid of same shape as input
    """
    # Get grid shape for reshaping later
    original_shape = grid.shape
    device = grid.device
    
    # Flatten the grid to make computations easier
    # Reshape to [N, 2] where N is the product of all dimensions except the last
    flat_shape = (-1, 2)
    grid_flat = grid.reshape(flat_shape)
    
    # Convert to homogeneous coordinates
    ones = torch.ones(grid_flat.shape[0], 1, device=device)
    homogeneous_grid = torch.cat([grid_flat, ones], dim=-1)
    
    # Compute inverse for sampling (where to get pixels from)
    try:
        inverse_matrix = torch.inverse(transform_matrix)
    except RuntimeError:
        # Handle singular matrix by adding small epsilon
        epsilon = 1e-6
        identity = torch.eye(3, device=device) * epsilon
        transform_matrix_reg = transform_matrix + identity
        inverse_matrix = torch.inverse(transform_matrix_reg)
    
    # Apply transformation
    transformed = torch.matmul(homogeneous_grid, inverse_matrix.T)
    
    # Normalize homogeneous coordinates (divide by w component)
    w = transformed[:, 2:3]
    mask = w != 0
    normalized = torch.zeros_like(transformed)
    
    # Apply normalization to x, y, z components where w != 0
    normalized[:, 0] = torch.where(mask.squeeze(-1), transformed[:, 0] / w.squeeze(-1), transformed[:, 0])
    normalized[:, 1] = torch.where(mask.squeeze(-1), transformed[:, 1] / w.squeeze(-1), transformed[:, 1])
    normalized[:, 2] = torch.where(mask.squeeze(-1), transformed[:, 2] / w.squeeze(-1), transformed[:, 2])
    
    # Get x, y components and reshape back to original grid shape
    transformed_grid = normalized[:, :2].reshape(original_shape)
    
    return transformed_grid


def warp_image(image, transform_matrix, mode="bilinear", padding_mode="zeros", align_corners=True):
    """
    Warp an image using a transformation matrix.
    
    Args:
        image: Input image tensor of shape [B, H, W, C]
        transform_matrix: 3x3 transformation matrix
        mode: Interpolation mode ('bilinear', 'nearest', or 'bicubic')
        padding_mode: Padding mode ('zeros', 'border', or 'reflection')
        align_corners: Whether to align corners in the grid_sample operation
        
    Returns:
        Warped image of same shape as input
    """
    # Get image dimensions
    B, H, W, C = image.shape
    device = image.device
    
    # Create sampling grid
    grid = create_sampling_grid(H, W, device)
    
    # Apply transformation to grid
    warped_grid = transform_grid(grid, transform_matrix)
    
    # Expand grid to batch dimension
    warped_grid = warped_grid.expand(B, -1, -1, -1)
    
    # Reorder image for grid_sample: [B, C, H, W]
    image_bchw = image.permute(0, 3, 1, 2)
    
    # Apply grid sampling
    warped_image = F.grid_sample(
        image_bchw, 
        warped_grid, 
        mode=mode, 
        padding_mode=padding_mode, 
        align_corners=align_corners
    )
    
    # Reorder back to [B, H, W, C]
    warped_image = warped_image.permute(0, 2, 3, 1)
    
    return warped_image


def warp_latent(latent, transform_matrix, mode="bilinear", padding_mode="zeros", align_corners=True):
    """
    Warp a latent tensor using a transformation matrix.
    
    Args:
        latent: Input latent dict with 'samples' tensor of shape [B, C, H, W]
        transform_matrix: 3x3 transformation matrix
        mode: Interpolation mode ('bilinear', 'nearest', or 'bicubic')
        padding_mode: Padding mode ('zeros', 'border', or 'reflection')
        align_corners: Whether to align corners in the grid_sample operation
        
    Returns:
        Warped latent in same format as input
    """
    # Get latent samples
    samples = latent["samples"]
    B, C, H, W = samples.shape
    device = samples.device
    
    # Create sampling grid
    grid = create_sampling_grid(H, W, device)
    
    # Apply transformation to grid
    warped_grid = transform_grid(grid, transform_matrix)
    
    # Expand grid to batch dimension
    warped_grid = warped_grid.expand(B, -1, -1, -1)
    
    # Apply grid sampling (latent is already in [B, C, H, W] format)
    warped_samples = F.grid_sample(
        samples, 
        warped_grid, 
        mode=mode, 
        padding_mode=padding_mode, 
        align_corners=align_corners
    )
    
    # Create new latent dict with warped samples
    warped_latent = latent.copy()
    warped_latent["samples"] = warped_samples
    
    return warped_latent 