import torch
import numpy as np
import cv2
import math
import logging
from typing import Union, List, Tuple

logger = logging.getLogger(__name__)

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Converts a hex color string (e.g., '#FF0000') to a BGR tuple (e.g., (0, 0, 255))."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        logger.warning(f"Invalid hex color format: {hex_color}. Using default black.")
        return (0, 0, 0)
    try:
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0]) # Convert RGB to BGR
    except ValueError:
        logger.warning(f"Invalid hex color value: {hex_color}. Using default black.")
        return (0, 0, 0)

# Node implementations will follow 

class CoordinateNormalizationNode:
    """
    Normalizes pixel coordinates (0 to Width/Height) to normalized coordinates (0.0 to 1.0)
    or denormalizes normalized coordinates back to pixel coordinates.
    Handles single coordinates or lists thereof.
    """
    CATEGORY = "Realtime Nodes/MediaPipe Vision/Common/Coordinates"
    FUNCTION = "normalize_or_denormalize"
    # Output type depends on mode, but ComfyUI needs fixed types.
    # We will output FLOAT for normalized, INT for pixel coords.
    # Since return types must be static, we use FLOAT and cast later if needed,
    # or potentially define two nodes. Let's return FLOAT for max flexibility.
    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("x_out", "y_out")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("FLOAT", {"default": 0.0, "forceInput": True, "tooltip": "X coordinate(s) (either pixel or normalized)"}),
                "y": ("FLOAT", {"default": 0.0, "forceInput": True, "tooltip": "Y coordinate(s) (either pixel or normalized)"}),
                "image_for_dimensions": ("IMAGE", {"tooltip": "Reference image to get Width and Height"}),
                "mode": (["Pixel Coords to Normalized", "Normalized Coords to Pixel"],),
            }
        }

    def normalize_or_denormalize(self, x: Union[float, List[float]], y: Union[float, List[float]],
                                 image_for_dimensions: torch.Tensor, mode: str):

        if image_for_dimensions.dim() != 4:
            logger.error("Input image must be BHWC format.")
            # Return neutral default based on input type
            return ([0.0] * len(x), [0.0] * len(y)) if isinstance(x, list) else (0.0, 0.0)

        _, height, width, _ = image_for_dimensions.shape

        is_x_list = isinstance(x, list)
        is_y_list = isinstance(y, list)

        if is_x_list != is_y_list:
            raise ValueError("Inputs x and y must be both floats or both lists.")

        if width == 0 or height == 0:
            logger.error("Image dimensions are zero.")
            return ([0.0] * len(x), [0.0] * len(y)) if is_x_list else (0.0, 0.0)

        if is_x_list:
            if len(x) != len(y):
                raise ValueError("Input lists x and y must have the same length.")
            if not x: # Empty lists
                return ([], [])

            x_out_list = []
            y_out_list = []
            for xi, yi in zip(x, y):
                if not isinstance(xi, (float, int)) or not isinstance(yi, (float, int)):
                     logger.warning(f"Invalid coordinate type in list (x={type(xi)}, y={type(yi)}). Skipping.")
                     x_out_list.append(0.0)
                     y_out_list.append(0.0)
                     continue

                if mode == "Pixel Coords to Normalized":
                    out_x = float(xi) / width
                    out_y = float(yi) / height
                elif mode == "Normalized Coords to Pixel":
                    # Output pixel coords as floats for consistency in return type
                    out_x = float(max(0, min(width - 1, xi * width)))
                    out_y = float(max(0, min(height - 1, yi * height)))
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                x_out_list.append(out_x)
                y_out_list.append(out_y)
            return (x_out_list, y_out_list)

        else: # Single float inputs
            if not isinstance(x, (float, int)) or not isinstance(y, (float, int)):
                 logger.error(f"Invalid coordinate type for single input (x={type(x)}, y={type(y)}).")
                 return (0.0, 0.0)

            if mode == "Pixel Coords to Normalized":
                out_x = float(x) / width
                out_y = float(y) / height
            elif mode == "Normalized Coords to Pixel":
                 out_x = float(max(0, min(width - 1, x * width)))
                 out_y = float(max(0, min(height - 1, y * height)))
            else:
                raise ValueError(f"Unknown mode: {mode}")
            return (out_x, out_y)


class DrawPointsNode:
    """
    Draws points (circles) on an image at specified normalized coordinates.
    Handles single coordinates or lists thereof. Operates batch-wise.
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
                "x": ("FLOAT", {"default": 0.5, "forceInput": True, "tooltip": "Normalized X coordinate(s) (0.0-1.0)"}),
                "y": ("FLOAT", {"default": 0.5, "forceInput": True, "tooltip": "Normalized Y coordinate(s) (0.0-1.0)"}),
                "radius": ("INT", {"default": 5, "min": 1, "max": 1024, "tooltip": "Radius of the points in pixels"}),
                "color_hex": ("STRING", {"default": "#FF0000", "tooltip": "Color in #RRGGBB format"}),
            }
        }

    def draw_points(self, image: torch.Tensor, x: Union[float, List[float]], y: Union[float, List[float]],
                    radius: int, color_hex: str):

        if image.dim() != 4:
            logger.error("Input image must be BHWC format.")
            return (image,)

        batch_size, height, width, channels = image.shape
        device = image.device
        dtype = image.dtype # Preserve original dtype

        is_x_list = isinstance(x, list)
        is_y_list = isinstance(y, list)

        if is_x_list != is_y_list:
            raise ValueError("Inputs x and y must be both floats or both lists.")

        # Wrap single inputs in lists for unified processing
        if not is_x_list:
            x_list = [x] * batch_size # Apply same point to all images in batch
            y_list = [y] * batch_size
        else:
            if len(x) != len(y):
                raise ValueError("Input lists x and y must have the same length.")
            if len(x) == 0: # Empty lists, return original image
                return (image,)
            # If list length matches batch size, use one point per image
            # If list length is 1, use that point for all images
            # Otherwise, it's ambiguous, maybe draw all points on all images?
            # Let's assume for now: if list, len must match batch_size or be 1.
            if len(x) != batch_size and len(x) != 1:
                 logger.warning(f"List input length ({len(x)}) doesn't match batch size ({batch_size}) or 1. Drawing all points on first image only.")
                 # Draw all points on the first image, return others unmodified.
                 x_list = x
                 y_list = y
                 batch_draw_limit = 1
            elif len(x) == 1:
                 x_list = x * batch_size
                 y_list = y * batch_size
                 batch_draw_limit = batch_size
            else: # len(x) == batch_size
                 x_list = x
                 y_list = y
                 batch_draw_limit = batch_size

        bgr_color = hex_to_bgr(color_hex)
        output_images = []

        # Copy image to avoid modifying input tensor directly
        image_copy = image.clone()

        for b in range(batch_size):
            # Get points for this batch item
            current_x = x_list[b] if len(x_list) == batch_size else x_list
            current_y = y_list[b] if len(y_list) == batch_size else y_list

            # Ensure current_x/y are iterable if we need to draw multiple points
            if not isinstance(current_x, list):
                current_x = [current_x]
                current_y = [current_y]

            if b >= batch_draw_limit and len(x) != 1 and len(x) != batch_size: # Only draw on first image if ambiguous len
                output_images.append(image_copy[b])
                continue

            # Process this image slice
            img_tensor_slice = image_copy[b].to(torch.float32) # Work with float32
            # Convert tensor (H, W, C) to numpy array (H, W, C) uint8
            img_np = (img_tensor_slice.cpu().numpy() * 255).astype(np.uint8)
            # Ensure it's contiguous
            img_np = np.ascontiguousarray(img_np)

            # Draw all points assigned to this image
            for xi, yi in zip(current_x, current_y):
                 if not isinstance(xi, (float, int)) or not isinstance(yi, (float, int)):
                    logger.warning(f"Invalid coordinate type at batch index {b} (x={type(xi)}, y={type(yi)}). Skipping point.")
                    continue

                 # Denormalize and clamp/clip
                 px = int(max(0, min(width - 1, xi * width)))
                 py = int(max(0, min(height - 1, yi * height)))

                 # Draw circle
                 cv2.circle(img_np, (px, py), radius, bgr_color, -1) # -1 for filled circle

            # Convert numpy array back to tensor (H, W, C)
            drawn_tensor_slice = torch.from_numpy(img_np.astype(np.float32) / 255.0)
            output_images.append(drawn_tensor_slice.to(device=device, dtype=dtype)) # Restore original device and dtype

        # Stack slices back into batch tensor (B, H, W, C)
        output_batch = torch.stack(output_images, dim=0)
        return (output_batch,)


class DrawLinesNode:
    """
    Draws lines on an image between specified pairs of normalized coordinates.
    Handles single coordinate pairs or lists thereof. Operates batch-wise.
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
                "x1": ("FLOAT", {"default": 0.25, "forceInput": True, "tooltip": "Normalized X coordinate(s) for start point (0.0-1.0)"}),
                "y1": ("FLOAT", {"default": 0.25, "forceInput": True, "tooltip": "Normalized Y coordinate(s) for start point (0.0-1.0)"}),
                "x2": ("FLOAT", {"default": 0.75, "forceInput": True, "tooltip": "Normalized X coordinate(s) for end point (0.0-1.0)"}),
                "y2": ("FLOAT", {"default": 0.75, "forceInput": True, "tooltip": "Normalized Y coordinate(s) for end point (0.0-1.0)"}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 1024, "tooltip": "Line thickness in pixels"}),
                "color_hex": ("STRING", {"default": "#00FF00", "tooltip": "Color in #RRGGBB format"}),
                "draw_endpoints": ("BOOLEAN", {"default": False, "tooltip": "Draw points at line endpoints"}),
                "point_radius": ("INT", {"default": 3, "min": 1, "max": 50, "tooltip": "Radius of endpoint points in pixels"}),
                "draw_label": ("BOOLEAN", {"default": False, "tooltip": "Draw a text label for the line"}),
                "label_text": ("STRING", {"default": "Line", "tooltip": "Text to display as the line label"}),
                "label_position": (["Midpoint", "Start", "End"], {"default": "Midpoint", "tooltip": "Where to place the label along the line"}),
                "font_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "Font scale factor for the label"}),
            }
        }

    def draw_lines(self, image: torch.Tensor,
                   x1: Union[float, List[float]], y1: Union[float, List[float]],
                   x2: Union[float, List[float]], y2: Union[float, List[float]],
                   thickness: int, color_hex: str, draw_endpoints: bool, point_radius: int,
                   draw_label: bool, label_text: str, label_position: str, font_scale: float):

        if image.dim() != 4:
            logger.error("Input image must be BHWC format.")
            return (image,)

        batch_size, height, width, channels = image.shape
        device = image.device
        dtype = image.dtype # Preserve original dtype

        # --- Input Type and Length Validation ---
        inputs = [x1, y1, x2, y2]
        is_list_input = isinstance(x1, list)
        if any(isinstance(inp, list) != is_list_input for inp in inputs):
             raise ValueError("All coordinate inputs (x1..y2) must be of the same type (all floats or all lists).")

        # Wrap single inputs in lists for unified processing
        if not is_list_input:
            # Apply same line to all images in batch
            x1_list, y1_list = [x1] * batch_size, [y1] * batch_size
            x2_list, y2_list = [x2] * batch_size, [y2] * batch_size
        else:
            list_len = len(x1)
            if any(len(inp) != list_len for inp in inputs if isinstance(inp, list)):
                 raise ValueError("If inputs are lists, all coordinate lists (x1..y2) must have the same length.")
            if list_len == 0:
                 return (image,) # No lines to draw

            # Determine how to map list inputs to batch
            if list_len != batch_size and list_len != 1:
                 logger.warning(f"List input length ({list_len}) doesn't match batch size ({batch_size}) or 1. Drawing all lines on first image only.")
                 x1_list, y1_list, x2_list, y2_list = x1, y1, x2, y2
                 batch_draw_limit = 1
            elif list_len == 1:
                 x1_list = x1 * batch_size
                 y1_list = y1 * batch_size
                 x2_list = x2 * batch_size
                 y2_list = y2 * batch_size
                 batch_draw_limit = batch_size
            else: # list_len == batch_size
                 x1_list, y1_list, x2_list, y2_list = x1, y1, x2, y2
                 batch_draw_limit = batch_size

        bgr_color = hex_to_bgr(color_hex)
        output_images = []

        # Copy image to avoid modifying input tensor directly
        image_copy = image.clone()

        for b in range(batch_size):
            # Get coordinate sets for this batch item
            # If original input was list[N] and N != batch_size and N != 1, draw all N lines on image b=0
            # If original input was list[1], draw that line on every image b
            # If original input was list[batch_size], draw line[b] on image[b]

            if b >= batch_draw_limit and list_len != 1 and list_len != batch_size:
                 output_images.append(image_copy[b])
                 continue

            # Process this image slice
            img_tensor_slice = image_copy[b].to(torch.float32)
            img_np = (img_tensor_slice.cpu().numpy() * 255).astype(np.uint8)
            img_np = np.ascontiguousarray(img_np)

            # Determine which lines to draw on this image
            indices_to_draw = []
            if list_len == batch_size:
                indices_to_draw = [b]
            elif list_len == 1:
                indices_to_draw = [0]
            elif b == 0: # Ambiguous case, draw all lines on first image
                indices_to_draw = range(list_len)

            for i in indices_to_draw:
                _x1, _y1 = x1_list[i], y1_list[i]
                _x2, _y2 = x2_list[i], y2_list[i]

                if not all(isinstance(c, (float, int)) for c in [_x1, _y1, _x2, _y2]):
                     logger.warning(f"Invalid coordinate type at batch index {b}, list index {i}. Skipping line.")
                     continue

                # Denormalize points
                px1 = int(max(0, min(width - 1, _x1 * width)))
                py1 = int(max(0, min(height - 1, _y1 * height)))
                px2 = int(max(0, min(width - 1, _x2 * width)))
                py2 = int(max(0, min(height - 1, _y2 * height)))

                # Draw line
                cv2.line(img_np, (px1, py1), (px2, py2), bgr_color, thickness)
                
                # Draw endpoint points if requested
                if draw_endpoints:
                    # Draw circles at both endpoints, reusing drawing logic from DrawPointsNode
                    cv2.circle(img_np, (px1, py1), point_radius, bgr_color, -1)
                    cv2.circle(img_np, (px2, py2), point_radius, bgr_color, -1)
                
                # Draw label if requested
                if draw_label and label_text:
                    # Determine label position based on user selection
                    if label_position == "Midpoint":
                        label_x = int((px1 + px2) / 2)
                        label_y = int((py1 + py2) / 2)
                    elif label_position == "Start":
                        label_x, label_y = px1, py1
                    elif label_position == "End":
                        label_x, label_y = px2, py2
                    else:  # Default to midpoint
                        label_x = int((px1 + px2) / 2)
                        label_y = int((py1 + py2) / 2)
                    
                    # Add a small offset for better visibility
                    label_y += 10
                    
                    # Get text size to center label properly
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_size = cv2.getTextSize(label_text, font, font_scale, 1)[0]
                    
                    # Center text horizontally
                    label_x = max(0, min(width - text_size[0], label_x - text_size[0] // 2))
                    
                    # Draw text with a slight outline/shadow for better visibility
                    # Draw dark outline
                    outline_color = (0, 0, 0)
                    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        cv2.putText(img_np, label_text, 
                                   (label_x + dx, label_y + dy), 
                                   font, font_scale, outline_color, 
                                   1, cv2.LINE_AA)
                    
                    # Draw main text
                    cv2.putText(img_np, label_text, 
                               (label_x, label_y), 
                               font, font_scale, bgr_color, 
                               1, cv2.LINE_AA)

            # Convert back to tensor
            drawn_tensor_slice = torch.from_numpy(img_np.astype(np.float32) / 255.0)
            output_images.append(drawn_tensor_slice.to(device=device, dtype=dtype))

        # Stack slices back into batch tensor
        output_batch = torch.stack(output_images, dim=0)
        return (output_batch,)

# Register nodes in NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "CoordinateNormalization": CoordinateNormalizationNode,
    "DrawPoints": DrawPointsNode,
    "DrawLines": DrawLinesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoordinateNormalization": "Coordinate Normalization",
    "DrawPoints": "Draw Points",
    "DrawLines": "Draw Lines",
}

# End of file 