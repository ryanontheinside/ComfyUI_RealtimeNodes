"""
Unified coordinate system for ComfyUI RealTimeNodes.

This module provides a centralized system for handling coordinates across different
spaces (normalized, pixel) with support for both single values and batches.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


class CoordinateSystem:
    """Unified coordinate system handling for coordinate space conversions."""

    # Constants for coordinate spaces
    NORMALIZED = "normalized"  # 0.0-1.0 range
    PIXEL = "pixel"  # Pixel coordinates

    @staticmethod
    def normalize(
        coords: Union[float, List[float], torch.Tensor],
        dimensions: Union[int, Tuple[int, ...]],
        input_space: str = PIXEL,
    ) -> Union[float, List[float], torch.Tensor]:
        """
        Normalize coordinates to 0.0-1.0 range.

        Args:
            coords: Coordinates to normalize
            dimensions: Dimensions to normalize by (width, height)
            input_space: Current coordinate space

        Returns:
            Normalized coordinates (0.0-1.0)
        """
        if input_space == CoordinateSystem.NORMALIZED:
            return coords

        # Handle different dimension specifications
        if isinstance(dimensions, tuple):
            dim_size = dimensions[0] if len(dimensions) == 1 else dimensions
        else:
            dim_size = dimensions

        # Handle different input types
        if isinstance(coords, list):
            if isinstance(dim_size, tuple):
                # Multiple dimensions
                if len(coords) % len(dim_size) != 0:
                    logger.warning(f"Coordinates length {len(coords)} not divisible by dimensions length {len(dim_size)}")
                result = []
                for i, c in enumerate(coords):
                    dim_idx = i % len(dim_size)
                    result.append(float(c) / dim_size[dim_idx])
                return result
            else:
                # Single dimension
                return [float(c) / dim_size for c in coords]
        elif isinstance(coords, torch.Tensor):
            if isinstance(dim_size, tuple):
                # Tensor with shape handling
                if coords.shape[-1] != len(dim_size):
                    raise ValueError(f"Tensor last dimension {coords.shape[-1]} doesn't match dimensions {len(dim_size)}")
                return coords.float() / torch.tensor(dim_size, device=coords.device)
            else:
                return coords.float() / dim_size
        else:
            return float(coords) / dim_size

    @staticmethod
    def denormalize(
        coords: Union[float, List[float], torch.Tensor],
        dimensions: Union[int, Tuple[int, ...]],
        output_space: str = PIXEL,
    ) -> Union[float, List[float], torch.Tensor]:
        """
        Convert normalized coordinates to target space.

        Args:
            coords: Normalized coordinates (0.0-1.0)
            dimensions: Target dimensions
            output_space: Target coordinate space

        Returns:
            Denormalized coordinates in target space
        """
        if output_space == CoordinateSystem.NORMALIZED:
            return coords

        # Handle different dimension specifications
        if isinstance(dimensions, tuple):
            dim_size = dimensions[0] if len(dimensions) == 1 else dimensions
        else:
            dim_size = dimensions

        # Handle different input types
        if isinstance(coords, list):
            if isinstance(dim_size, tuple):
                # Multiple dimensions
                if len(coords) % len(dim_size) != 0:
                    logger.warning(f"Coordinates length {len(coords)} not divisible by dimensions length {len(dim_size)}")
                result = []
                for i, c in enumerate(coords):
                    dim_idx = i % len(dim_size)
                    result.append(float(max(0, min(dim_size[dim_idx] - 1, c * dim_size[dim_idx]))))
                return result
            else:
                # Single dimension
                return [float(max(0, min(dim_size - 1, c * dim_size))) for c in coords]
        elif isinstance(coords, torch.Tensor):
            if isinstance(dim_size, tuple):
                # Tensor with shape handling
                if coords.shape[-1] != len(dim_size):
                    raise ValueError(f"Tensor last dimension {coords.shape[-1]} doesn't match dimensions {len(dim_size)}")
                dim_tensor = torch.tensor(dim_size, device=coords.device)
                return torch.clamp(coords * dim_tensor, 0, dim_tensor - 1)
            else:
                return torch.clamp(coords * dim_size, 0, dim_size - 1)
        else:
            return float(max(0, min(dim_size - 1, coords * dim_size)))

    @staticmethod
    def convert(
        coords: Union[float, List[float], torch.Tensor],
        dimensions: Union[int, Tuple[int, ...]],
        from_space: str,
        to_space: str,
    ) -> Union[float, List[float], torch.Tensor]:
        """
        Convert between coordinate spaces.

        Args:
            coords: Coordinates to convert
            dimensions: Target dimensions
            from_space: Source coordinate space
            to_space: Target coordinate space

        Returns:
            Converted coordinates
        """
        if from_space == to_space:
            return coords

        # First normalize if needed
        if from_space != CoordinateSystem.NORMALIZED:
            normalized = CoordinateSystem.normalize(coords, dimensions, from_space)
        else:
            normalized = coords

        # Then denormalize to target space if needed
        if to_space != CoordinateSystem.NORMALIZED:
            return CoordinateSystem.denormalize(normalized, dimensions, to_space)
        else:
            return normalized

    @staticmethod
    def get_dimensions_from_tensor(tensor: torch.Tensor) -> Tuple[int, int]:
        """
        Extract dimensions from tensor in BHWC format.

        Args:
            tensor: Input tensor in BHWC format

        Returns:
            Tuple of (width, height)
        """
        if tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor (BHWC), got shape {tensor.shape}")

        _, height, width, _ = tensor.shape
        return (width, height)


class Point:
    """
    Represents a 2D or 3D point with coordinate space awareness.
    """

    def __init__(self, x: float, y: float, z: Optional[float] = None, space: str = CoordinateSystem.NORMALIZED):
        """
        Initialize a point with coordinates and space information.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate (optional)
            space: Coordinate space
        """
        self.x = x
        self.y = y
        self.z = z
        self.space = space
        self.is_3d = z is not None

    def to_normalized(self, dimensions: Tuple[int, ...]) -> "Point":
        """
        Convert point to normalized space.

        Args:
            dimensions: Dimensions for conversion

        Returns:
            New point in normalized space
        """
        if self.space == CoordinateSystem.NORMALIZED:
            return self

        norm_x = CoordinateSystem.normalize(self.x, dimensions[0], self.space)
        norm_y = CoordinateSystem.normalize(self.y, dimensions[1], self.space)
        norm_z = None

        if self.is_3d and len(dimensions) > 2:
            norm_z = CoordinateSystem.normalize(self.z, dimensions[2], self.space)

        return Point(norm_x, norm_y, norm_z, CoordinateSystem.NORMALIZED)

    def to_pixel(self, dimensions: Tuple[int, ...]) -> "Point":
        """
        Convert point to pixel space.

        Args:
            dimensions: Dimensions for conversion

        Returns:
            New point in pixel space
        """
        if self.space == CoordinateSystem.PIXEL:
            return self

        # First normalize if not already
        if self.space != CoordinateSystem.NORMALIZED:
            normalized = self.to_normalized(dimensions)
        else:
            normalized = self

        pixel_x = CoordinateSystem.denormalize(normalized.x, dimensions[0], CoordinateSystem.PIXEL)
        pixel_y = CoordinateSystem.denormalize(normalized.y, dimensions[1], CoordinateSystem.PIXEL)
        pixel_z = None

        if self.is_3d and len(dimensions) > 2:
            pixel_z = CoordinateSystem.denormalize(normalized.z, dimensions[2], CoordinateSystem.PIXEL)

        return Point(pixel_x, pixel_y, pixel_z, CoordinateSystem.PIXEL)

    def to_space(self, dimensions: Tuple[int, ...], target_space: str) -> "Point":
        """
        Convert point to target space.

        Args:
            dimensions: Dimensions for conversion
            target_space: Target coordinate space

        Returns:
            New point in target space
        """
        if self.space == target_space:
            return self

        if target_space == CoordinateSystem.NORMALIZED:
            return self.to_normalized(dimensions)
        elif target_space == CoordinateSystem.PIXEL:
            return self.to_pixel(dimensions)
        else:
            raise ValueError(f"Unsupported target space: {target_space}")

    def to_tuple(self, include_z: bool = False) -> Tuple[float, ...]:
        """
        Convert point to tuple representation.

        Args:
            include_z: Whether to include Z coordinate if available

        Returns:
            Tuple of coordinates
        """
        if include_z and self.is_3d:
            return (self.x, self.y, self.z)
        else:
            return (self.x, self.y)

    def to_list(self, include_z: bool = False) -> List[float]:
        """
        Convert point to list representation.

        Args:
            include_z: Whether to include Z coordinate if available

        Returns:
            List of coordinates
        """
        return list(self.to_tuple(include_z))

    def __repr__(self) -> str:
        """String representation of the point."""
        if self.is_3d:
            return f"Point({self.x}, {self.y}, {self.z}, space='{self.space}')"
        else:
            return f"Point({self.x}, {self.y}, space='{self.space}')"


class PointList:
    """
    Represents a list of 2D or 3D points with coordinate space awareness.
    Useful for handling batches of points or landmark lists.
    """

    def __init__(self, points: List[Point] = None, space: str = CoordinateSystem.NORMALIZED):
        """
        Initialize a point list.

        Args:
            points: List of Point objects
            space: Coordinate space for all points
        """
        self.points = points or []
        self.space = space

    @classmethod
    def from_coordinates(
        cls,
        x_coords: List[float],
        y_coords: List[float],
        z_coords: Optional[List[float]] = None,
        space: str = CoordinateSystem.NORMALIZED,
    ) -> "PointList":
        """
        Create a PointList from separate coordinate lists.

        Args:
            x_coords: List of X coordinates
            y_coords: List of Y coordinates
            z_coords: List of Z coordinates (optional)
            space: Coordinate space

        Returns:
            New PointList object
        """
        if len(x_coords) != len(y_coords):
            raise ValueError(f"Coordinate lists must have same length: x={len(x_coords)}, y={len(y_coords)}")

        has_z = z_coords is not None
        if has_z and len(z_coords) != len(x_coords):
            raise ValueError(f"Z coordinates must match length of X/Y: x={len(x_coords)}, z={len(z_coords)}")

        points = []
        for i in range(len(x_coords)):
            z = z_coords[i] if has_z else None
            points.append(Point(x_coords[i], y_coords[i], z, space))

        return cls(points, space)

    def to_normalized(self, dimensions: Tuple[int, ...]) -> "PointList":
        """
        Convert all points to normalized space.

        Args:
            dimensions: Dimensions for conversion

        Returns:
            New PointList in normalized space
        """
        if self.space == CoordinateSystem.NORMALIZED:
            return self

        normalized_points = [point.to_normalized(dimensions) for point in self.points]
        return PointList(normalized_points, CoordinateSystem.NORMALIZED)

    def to_pixel(self, dimensions: Tuple[int, ...]) -> "PointList":
        """
        Convert all points to pixel space.

        Args:
            dimensions: Dimensions for conversion

        Returns:
            New PointList in pixel space
        """
        if self.space == CoordinateSystem.PIXEL:
            return self

        pixel_points = [point.to_pixel(dimensions) for point in self.points]
        return PointList(pixel_points, CoordinateSystem.PIXEL)

    def to_space(self, dimensions: Tuple[int, ...], target_space: str) -> "PointList":
        """
        Convert all points to target space.

        Args:
            dimensions: Dimensions for conversion
            target_space: Target coordinate space

        Returns:
            New PointList in target space
        """
        if self.space == target_space:
            return self

        if target_space == CoordinateSystem.NORMALIZED:
            return self.to_normalized(dimensions)
        elif target_space == CoordinateSystem.PIXEL:
            return self.to_pixel(dimensions)
        else:
            raise ValueError(f"Unsupported target space: {target_space}")

    def get_x_list(self) -> List[float]:
        """Get list of X coordinates."""
        return [point.x for point in self.points]

    def get_y_list(self) -> List[float]:
        """Get list of Y coordinates."""
        return [point.y for point in self.points]

    def get_z_list(self) -> List[Optional[float]]:
        """Get list of Z coordinates (may contain None values)."""
        return [point.z for point in self.points]

    def __len__(self) -> int:
        """Get number of points in the list."""
        return len(self.points)

    def __getitem__(self, idx: int) -> Point:
        """Access point by index."""
        return self.points[idx]
