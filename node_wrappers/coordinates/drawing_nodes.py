import logging
from typing import List, Union

import torch

from ...src.coordinates import CoordinateSystem, DrawingEngine
from ...src.coordinates import coordinate_utils, drawing_utils

logger = logging.getLogger(__name__)


class RTDrawPointsNode:
    """
    High-performance point drawing node optimized for real-time applications.
    Draws points (circles) on images with minimal overhead.
    """

    CATEGORY = "Realtime Nodes/MediaPipe Vision/Common/Drawing"
    FUNCTION = "draw_points"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": (
                    "FLOAT",
                    {"default": 0.5, "forceInput": True, "tooltip": "X coordinate(s) (0.0-1.0 normalized by default)"},
                ),
                "y": (
                    "FLOAT",
                    {"default": 0.5, "forceInput": True, "tooltip": "Y coordinate(s) (0.0-1.0 normalized by default)"},
                ),
                "radius": ("INT", {"default": 5, "min": 1, "max": 100, "tooltip": "Radius of points in pixels"}),
                "color_hex": ("STRING", {"default": "#FF0000", "tooltip": "Color in #RRGGBB format"}),
            },
            "optional": {
                "is_normalized": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Whether coordinates are normalized (0-1) or pixel values"},
                ),
                "batch_mapping": (
                    ["broadcast", "one-to-one", "all-on-first"],
                    {"default": "broadcast", "tooltip": "How to map coords to batch images"},
                ),
            },
        }

    def draw_points(
        self,
        image: torch.Tensor,
        x: Union[float, List[float]],
        y: Union[float, List[float]],
        radius: int,
        color_hex: str,
        is_normalized: bool = True,
        batch_mapping: str = "broadcast",
    ):
        """Draw points efficiently using the DrawingEngine."""
        # Setup drawing engine and coordinate system
        drawing_engine, space, dimensions = drawing_utils.setup_drawing_engine_and_coordinates(image, is_normalized)
        
        return (
            drawing_engine.draw_points(
                image=image,
                points_x=x,
                points_y=y,
                radius=radius,
                color=color_hex,
                is_normalized=is_normalized,
                batch_mapping=batch_mapping,
            ),
        )


class RTDrawLinesNode:
    """
    High-performance line drawing node optimized for real-time applications.
    Draws lines on images with minimal overhead.
    """

    CATEGORY = "Realtime Nodes/MediaPipe Vision/Common/Drawing"
    FUNCTION = "draw_lines"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x1": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "forceInput": True,
                        "tooltip": "Start X coordinate(s) (0.0-1.0 normalized by default)",
                    },
                ),
                "y1": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "forceInput": True,
                        "tooltip": "Start Y coordinate(s) (0.0-1.0 normalized by default)",
                    },
                ),
                "x2": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "forceInput": True,
                        "tooltip": "End X coordinate(s) (0.0-1.0 normalized by default)",
                    },
                ),
                "y2": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "forceInput": True,
                        "tooltip": "End Y coordinate(s) (0.0-1.0 normalized by default)",
                    },
                ),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 20, "tooltip": "Line thickness in pixels"}),
                "color_hex": ("STRING", {"default": "#00FF00", "tooltip": "Color in #RRGGBB format"}),
            },
            "optional": {
                "is_normalized": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Whether coordinates are normalized (0-1) or pixel values"},
                ),
                "batch_mapping": (["broadcast", "one-to-one", "all-on-first"], {"default": "broadcast"}),
                "draw_endpoints": ("BOOLEAN", {"default": False, "tooltip": "Draw circles at line endpoints"}),
                "point_radius": (
                    "INT",
                    {"default": 3, "min": 1, "max": 20, "tooltip": "Radius of endpoint circles in pixels"},
                ),
                "draw_label": ("BOOLEAN", {"default": False, "tooltip": "Draw text label on line"}),
                "label_prefix": ("STRING", {"default": "Line", "tooltip": "Prefix for label"}),
                "label_values": ("STRING", {"default": "", "tooltip": "Values for label"}),
                "label_position": (["Midpoint", "Start", "End"], {"default": "Midpoint"}),
                "font_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1}),
                "batch_value_mode": (["Index-based", "First value only"], {"default": "Index-based", "tooltip": "How to apply values across batch"}),
            },
        }

    def draw_lines(
        self,
        image: torch.Tensor,
        x1: Union[float, List[float]],
        y1: Union[float, List[float]],
        x2: Union[float, List[float]],
        y2: Union[float, List[float]],
        thickness: int,
        color_hex: str,
        is_normalized: bool = True,
        batch_mapping: str = "broadcast",
        draw_endpoints: bool = False,
        point_radius: int = 3,
        draw_label: bool = False,
        label_prefix: str = "Line",
        label_values: str = "",
        label_position: str = "Midpoint",
        font_scale: float = 0.5,
        batch_value_mode: str = "Index-based",
    ):
        """Draw lines efficiently using the DrawingEngine."""
        # Setup drawing engine and coordinate system
        drawing_engine, space, dimensions = drawing_utils.setup_drawing_engine_and_coordinates(image, is_normalized)

        # Parse the values list
        values_list = coordinate_utils.parse_label_values(label_values)
        
        # Setup batch drawing
        batch_size, output_batch = drawing_utils.setup_batch_drawing(image)
        
        # Standardize inputs to lists
        x1_list, y1_list, x2_list, y2_list = coordinate_utils.standardize_inputs_to_lists(x1, y1, x2, y2)
        
        # Process each batch item
        for b in range(batch_size):
            # Handle different batch mapping strategies for coordinates
            coord_lists = [x1_list, y1_list, x2_list, y2_list]
            should_skip, batch_coords = drawing_utils.handle_batch_mapping_coordinates(b, batch_mapping, coord_lists, batch_size)
            
            if should_skip:
                continue
                
            if batch_mapping == "one-to-one":
                batch_x1, batch_y1, batch_x2, batch_y2 = batch_coords
            else:
                # Broadcast: all coordinates on all batches
                batch_x1, batch_y1, batch_x2, batch_y2 = batch_coords
            
            # Determine the label for this batch
            current_label = coordinate_utils.determine_batch_label(values_list, b, label_prefix, batch_value_mode)
            
            # Create a single-item tensor for this batch
            single_image = output_batch[b:b+1]
            
            # Process this batch item
            result = drawing_engine.draw_lines(
                image=single_image,
                x1=batch_x1,
                y1=batch_y1,
                x2=batch_x2,
                y2=batch_y2,
                thickness=thickness,
                color=color_hex,
                is_normalized=is_normalized,
                batch_mapping="broadcast",  # Always broadcast within the single item
                draw_endpoints=draw_endpoints,
                point_radius=point_radius,
                draw_label=draw_label,
                label_text=current_label,
                label_position=label_position,
                font_scale=font_scale,
            )
            
            # Update the output batch
            output_batch[b:b+1] = result
        
        return (output_batch,)


class RTDrawPolygonNode:
    """
    High-performance polygon drawing node optimized for real-time applications.
    Draws polygons on images with minimal overhead.
    """

    CATEGORY = "Realtime Nodes/MediaPipe Vision/Common/Drawing"
    FUNCTION = "draw_polygon"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "points_x": (
                    "FLOAT",
                    {
                        "default": [0.25, 0.75, 0.5],
                        "forceInput": True,
                        "tooltip": "List of X coordinates for polygon vertices",
                    },
                ),
                "points_y": (
                    "FLOAT",
                    {
                        "default": [0.25, 0.25, 0.75],
                        "forceInput": True,
                        "tooltip": "List of Y coordinates for polygon vertices",
                    },
                ),
                "color_hex": ("STRING", {"default": "#0000FF", "tooltip": "Color in #RRGGBB format"}),
                "thickness": (
                    "INT",
                    {"default": 2, "min": 1, "max": 20, "tooltip": "Line thickness for polygon outline"},
                ),
            },
            "optional": {
                "fill": ("BOOLEAN", {"default": False, "tooltip": "Whether to fill the polygon"}),
                "is_normalized": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Whether coordinates are normalized (0-1) or pixel values"},
                ),
                "batch_mapping": (["broadcast", "one-to-one", "all-on-first"], {"default": "broadcast"}),
                "draw_vertices": ("BOOLEAN", {"default": False, "tooltip": "Draw points at polygon vertices"}),
                "vertex_radius": ("INT", {"default": 3, "min": 1, "max": 20, "tooltip": "Radius of vertex points"}),
                "draw_label": ("BOOLEAN", {"default": False, "tooltip": "Draw text label on polygon"}),
                "label_prefix": ("STRING", {"default": "Polygon", "tooltip": "Prefix for label"}),
                "label_values": ("STRING", {"default": "", "tooltip": "Values for label"}),
                "font_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1}),
                "batch_value_mode": (["Index-based", "First value only"], {"default": "Index-based", "tooltip": "How to apply values across batch"}),
            },
        }

    def draw_polygon(
        self,
        image: torch.Tensor,
        points_x: List[float],
        points_y: List[float],
        color_hex: str,
        thickness: int,
        fill: bool = False,
        is_normalized: bool = True,
        batch_mapping: str = "broadcast",
        draw_vertices: bool = False,
        vertex_radius: int = 3,
        draw_label: bool = False,
        label_prefix: str = "Polygon",
        label_values: str = "",
        font_scale: float = 0.5,
        batch_value_mode: str = "Index-based",
    ):
        """Draw polygon efficiently using the DrawingEngine."""
        # Setup drawing engine and coordinate system
        drawing_engine, space, dimensions = drawing_utils.setup_drawing_engine_and_coordinates(image, is_normalized)

        # Parse the values list
        values_list = coordinate_utils.parse_label_values(label_values)
        
        # Setup batch drawing
        batch_size, output_batch = drawing_utils.setup_batch_drawing(image)
        
        # Process each batch item
        for b in range(batch_size):
            # Skip processing based on mapping strategy
            if batch_mapping == "all-on-first" and b > 0:
                continue
                
            # Determine the label for this batch
            current_label = coordinate_utils.determine_batch_label(values_list, b, label_prefix, batch_value_mode)
            
            # Create a single-item tensor for this batch
            single_image = output_batch[b:b+1]
            
            # Process this batch item
            result = drawing_engine.draw_polygon(
                image=single_image,
                points_x=points_x,
                points_y=points_y,
                color=color_hex,
                thickness=thickness,
                fill=fill,
                is_normalized=is_normalized,
                batch_mapping="broadcast",  # Always broadcast within the single item
                draw_vertices=draw_vertices,
                vertex_radius=vertex_radius,
                draw_label=draw_label,
                label_text=current_label,
                font_scale=font_scale,
            )
            
            # Update the output batch
            output_batch[b:b+1] = result
        
        return (output_batch,)


class RTCoordinateConverterNode:
    """
    Fast coordinate conversion between normalized (0-1) and pixel space.
    Optimized for real-time applications with minimal overhead.
    """

    CATEGORY = "Realtime Nodes/MediaPipe Vision/Common/Coordinates"
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
                "mode": (["Pixel to Normalized", "Normalized to Pixel"],),
            }
        }

    def convert_coordinates(
        self, x: Union[float, List[float]], y: Union[float, List[float]], image_for_dimensions: torch.Tensor, mode: str
    ):
        """Convert coordinates using the unified coordinate system."""
        # Setup coordinate system
        space, dimensions = coordinate_utils.get_coordinate_space_and_dimensions(mode == "normalized", image_for_dimensions)
        
        # Convert coordinates based on mode
        if mode == "to_pixel":
            x_out = CoordinateSystem.denormalize(x, dimensions[0], CoordinateSystem.PIXEL)
            y_out = CoordinateSystem.denormalize(y, dimensions[1], CoordinateSystem.PIXEL)
        elif mode == "to_normalized":
            x_out = CoordinateSystem.normalize(x, dimensions[0], CoordinateSystem.PIXEL)
            y_out = CoordinateSystem.normalize(y, dimensions[1], CoordinateSystem.PIXEL)
        else:
            # No conversion needed
            x_out = x
            y_out = y
            
        return (x_out, y_out)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "RTDrawPoints": RTDrawPointsNode,
    "RTDrawLines": RTDrawLinesNode,
    "RTDrawPolygon": RTDrawPolygonNode,
    "RTCoordinateConverter": RTCoordinateConverterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RTDrawPoints": "Draw Points (Fast)",
    "RTDrawLines": "Draw Lines (Fast)",
    "RTDrawPolygon": "Draw Polygon (Fast)",
    "RTCoordinateConverter": "Coordinate Converter (Fast)",
}
