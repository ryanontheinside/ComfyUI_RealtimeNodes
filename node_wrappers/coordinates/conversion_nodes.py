"""
Node wrappers for coordinate conversion in ComfyUI.

These nodes provide user-friendly interfaces for the coordinate system functionality.
"""

import logging

from ...src.coordinates import CoordinateSystem

logger = logging.getLogger(__name__)


class CoordinateConverterNode:
    """
    Fast coordinate conversion between different coordinate spaces.
    """

    CATEGORY = "Realtime Nodes/Coordinates"
    FUNCTION = "convert_coordinates"
    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("x_out", "y_out")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("FLOAT", {"default": 0.0, "forceInput": True, "tooltip": "X coordinate(s) to convert"}),
                "y": ("FLOAT", {"default": 0.0, "forceInput": True, "tooltip": "Y coordinate(s) to convert"}),
                "image_for_dimensions": ("IMAGE", {"tooltip": "Reference image for dimensions"}),
                "from_space": (["pixel", "normalized"],),
                "to_space": (["pixel", "normalized"],),
            }
        }

    def convert_coordinates(self, x, y, image_for_dimensions, from_space, to_space):
        """Convert coordinates using the unified coordinate system."""
        if image_for_dimensions.dim() != 4:
            logger.error("Input image must be BHWC format.")
            return ([0.0] * len(x), [0.0] * len(y)) if isinstance(x, list) else (0.0, 0.0)

        try:
            # Get dimensions from image
            dimensions = CoordinateSystem.get_dimensions_from_tensor(image_for_dimensions)

            # Map string inputs to CoordinateSystem constants
            space_map = {
                "pixel": CoordinateSystem.PIXEL,
                "normalized": CoordinateSystem.NORMALIZED,
            }

            from_space_const = space_map[from_space]
            to_space_const = space_map[to_space]

            # Convert coordinates
            x_out = CoordinateSystem.convert(x, dimensions[0], from_space_const, to_space_const)
            y_out = CoordinateSystem.convert(y, dimensions[1], from_space_const, to_space_const)

            return (x_out, y_out)
        except Exception as e:
            logger.error(f"Error in coordinate conversion: {e}")
            return ([0.0] * len(x), [0.0] * len(y)) if isinstance(x, list) else (0.0, 0.0)


class Point2DNode:
    """
    Creates a 2D point with coordinate space awareness.
    """

    CATEGORY = "Realtime Nodes/Coordinates"
    FUNCTION = "create_point"
    RETURN_TYPES = ("POINT",)
    RETURN_NAMES = ("point",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "X coordinate"}),
                "y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Y coordinate"}),
                "space": (["normalized", "pixel"],),
            },
            "optional": {
                "image_for_dimensions": (
                    "IMAGE",
                    {"tooltip": "Reference image for dimensions (required for pixel space)"},
                ),
            },
        }

    def create_point(self, x, y, space, image_for_dimensions=None):
        """Create a Point object."""
        from ...src.coordinates import Point

        space_map = {
            "pixel": CoordinateSystem.PIXEL,
            "normalized": CoordinateSystem.NORMALIZED,
        }

        space_const = space_map[space]

        # Validate dimensions if using pixel space
        if space == "pixel" and image_for_dimensions is None:
            logger.warning("Pixel space selected but no image provided for dimensions. Using normalized space.")
            space_const = CoordinateSystem.NORMALIZED

        return (Point(x, y, None, space_const),)


class PointListNode:
    """
    Creates a list of points from coordinate lists.
    """

    CATEGORY = "Realtime Nodes/Coordinates"
    FUNCTION = "create_point_list"
    RETURN_TYPES = ("POINT_LIST",)
    RETURN_NAMES = ("point_list",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x_coords": (
                    "FLOAT",
                    {"default": [0.25, 0.5, 0.75], "forceInput": True, "tooltip": "List of X coordinates"},
                ),
                "y_coords": (
                    "FLOAT",
                    {"default": [0.25, 0.5, 0.75], "forceInput": True, "tooltip": "List of Y coordinates"},
                ),
                "space": (["normalized", "pixel"],),
            },
            "optional": {
                "z_coords": (
                    "FLOAT",
                    {"default": None, "forceInput": True, "tooltip": "List of Z coordinates (optional)"},
                ),
                "image_for_dimensions": (
                    "IMAGE",
                    {"tooltip": "Reference image for dimensions (required for pixel space)"},
                ),
            },
        }

    def create_point_list(self, x_coords, y_coords, space, z_coords=None, image_for_dimensions=None):
        """Create a PointList object."""
        from ...src.coordinates import PointList

        space_map = {
            "pixel": CoordinateSystem.PIXEL,
            "normalized": CoordinateSystem.NORMALIZED,
        }

        space_const = space_map[space]

        # Validate dimensions if using pixel space
        if space == "pixel" and image_for_dimensions is None:
            logger.warning("Pixel space selected but no image provided for dimensions. Using normalized space.")
            space_const = CoordinateSystem.NORMALIZED

        # Ensure inputs are lists
        if not isinstance(x_coords, list):
            x_coords = [x_coords]
        if not isinstance(y_coords, list):
            y_coords = [y_coords]
        if z_coords is not None and not isinstance(z_coords, list):
            z_coords = [z_coords]

        return (PointList.from_coordinates(x_coords, y_coords, z_coords, space_const),)


# Node mapping for ComfyUI
NODE_CLASS_MAPPINGS = {
    "CoordinateConverter": CoordinateConverterNode,
    "Point2D": Point2DNode,
    "PointList": PointListNode,
}

# Display names for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "CoordinateConverter": "Coordinate Converter",
    "Point2D": "Create 2D Point",
    "PointList": "Create Point List",
}
