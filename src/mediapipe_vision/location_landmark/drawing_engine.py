import torch
import numpy as np
import cv2
from typing import Union, List, Tuple, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class DrawingEngine:
    """High-performance drawing engine for real-time applications.
    Optimized for minimal tensor-numpy conversions and efficient batch operations.
    """
    
    @staticmethod
    def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
        """Converts hex color to BGR tuple with validation."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return (0, 0, 0)  # Default black
        try:
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return (rgb[2], rgb[1], rgb[0])  # RGB to BGR
        except ValueError:
            return (0, 0, 0)

    @staticmethod
    def normalize_coords(coords: Union[float, List[float]], dim_size: int) -> Union[float, List[float]]:
        """Normalize pixel coordinates to 0.0-1.0 range."""
        if isinstance(coords, list):
            return [float(c) / dim_size for c in coords]
        return float(coords) / dim_size

    @staticmethod
    def denormalize_coords(coords: Union[float, List[float]], dim_size: int) -> Union[float, List[float]]:
        """Convert normalized coordinates to pixel space."""
        if isinstance(coords, list):
            return [float(max(0, min(dim_size - 1, c * dim_size))) for c in coords]
        return float(max(0, min(dim_size - 1, coords * dim_size)))

    @staticmethod
    def prep_batch_image(image: torch.Tensor, batch_idx: int) -> Tuple[np.ndarray, torch.device, torch.dtype]:
        """Fast preparation of single image from batch for drawing - minimal conversion."""
        if image.dim() != 4:
            raise ValueError("Input image must be BHWC format")
            
        device, dtype = image.device, image.dtype
        # Get single image, convert to float32 for processing
        img_tensor = image[batch_idx].to(torch.float32)
        # Single efficient conversion to numpy
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        # Ensure contiguous for CV2 operations
        return np.ascontiguousarray(img_np), device, dtype

    @staticmethod
    def tensor_from_numpy(img_np: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Fast conversion from numpy back to tensor with proper device/dtype."""
        return torch.from_numpy(img_np.astype(np.float32) / 255.0).to(device=device, dtype=dtype)

    def draw_points(self, 
                   image: torch.Tensor,
                   points_x: Union[float, List[float]],
                   points_y: Union[float, List[float]],
                   radius: int = 5,
                   color: str = "#FF0000",
                   is_normalized: bool = True,
                   batch_mapping: str = "broadcast") -> torch.Tensor:
        """
        Draw points on images efficiently.
        
        Args:
            image: BHWC tensor
            points_x/y: Coordinate values (normalized 0-1 or pixel)
            radius: Point radius in pixels
            color: Hex color
            is_normalized: If True, coords are 0-1 normalized
            batch_mapping: How to map points to batch ("broadcast", "one-to-one", "all-on-first")
            
        Returns:
            Modified image tensor
        """
        batch_size, height, width, _ = image.shape
        output_batch = image.clone()  # Avoid modifying input
        bgr_color = self.hex_to_bgr(color)
        
        # Standardize input to lists for unified processing
        x_list = points_x if isinstance(points_x, list) else [points_x]
        y_list = points_y if isinstance(points_y, list) else [points_y]
        
        # Determine batch mapping strategy
        points_len = len(x_list)
        if batch_mapping == "one-to-one" and points_len != batch_size and points_len != 1:
            logger.warning(f"One-to-one mapping requested but point list length ({points_len}) != batch size ({batch_size})")
            batch_mapping = "all-on-first"
            
        # Process each image in batch
        for b in range(batch_size):
            # Skip processing based on mapping strategy
            if batch_mapping == "all-on-first" and b > 0:
                continue
                
            # Get points for this batch item based on mapping strategy
            if batch_mapping == "one-to-one" and points_len == batch_size:
                # One point per image
                curr_x = [x_list[b]]
                curr_y = [y_list[b]]
            elif batch_mapping == "broadcast" or (batch_mapping == "one-to-one" and points_len == 1):
                # Same point(s) on all images
                curr_x = x_list
                curr_y = y_list
            else:  # "all-on-first"
                # All points on first image only
                curr_x = x_list
                curr_y = y_list
                
            # Skip if no points to draw
            if not curr_x:
                continue
                
            # Single efficient conversion to numpy
            img_np, device, dtype = self.prep_batch_image(output_batch, b)
            
            # Draw all points for this image
            for xi, yi in zip(curr_x, curr_y):
                if not isinstance(xi, (float, int)) or not isinstance(yi, (float, int)):
                    continue
                    
                # Convert to pixel coordinates if needed
                if is_normalized:
                    px = int(max(0, min(width - 1, xi * width)))
                    py = int(max(0, min(height - 1, yi * height)))
                else:
                    px = int(max(0, min(width - 1, xi)))
                    py = int(max(0, min(height - 1, yi)))
                    
                # Draw circle
                cv2.circle(img_np, (px, py), radius, bgr_color, -1)
                
            # Convert back to tensor and update batch
            output_batch[b] = self.tensor_from_numpy(img_np, device, dtype)
            
        return output_batch
        
    def draw_lines(self,
                  image: torch.Tensor,
                  x1: Union[float, List[float]],
                  y1: Union[float, List[float]],
                  x2: Union[float, List[float]],
                  y2: Union[float, List[float]],
                  thickness: int = 2,
                  color: str = "#00FF00",
                  is_normalized: bool = True,
                  batch_mapping: str = "broadcast",
                  draw_endpoints: bool = False,
                  point_radius: int = 3,
                  draw_label: bool = False,
                  label_text: str = "",
                  label_position: str = "Midpoint",
                  font_scale: float = 0.5) -> torch.Tensor:
        """
        Draw lines on images efficiently.
        
        Args:
            image: BHWC tensor
            x1/y1/x2/y2: Line endpoint coordinates
            thickness: Line thickness in pixels
            color: Hex color
            is_normalized: If True, coords are 0-1 normalized
            batch_mapping: How to map lines to batch
            draw_endpoints: Whether to draw circles at endpoints
            Other params: Styling options
            
        Returns:
            Modified image tensor
        """
        batch_size, height, width, _ = image.shape
        output_batch = image.clone()
        bgr_color = self.hex_to_bgr(color)
        
        # Standardize input to lists
        x1_list = x1 if isinstance(x1, list) else [x1]
        y1_list = y1 if isinstance(y1, list) else [y1]
        x2_list = x2 if isinstance(x2, list) else [x2]
        y2_list = y2 if isinstance(y2, list) else [y2]
        
        # Validate input lengths
        line_len = len(x1_list)
        if not all(len(coord) == line_len for coord in [y1_list, x2_list, y2_list]):
            raise ValueError("All coordinate lists must have the same length")
            
        # Determine batch mapping
        if batch_mapping == "one-to-one" and line_len != batch_size and line_len != 1:
            logger.warning(f"One-to-one mapping requested but line list length ({line_len}) != batch size ({batch_size})")
            batch_mapping = "all-on-first"
            
        # Process each image in batch
        for b in range(batch_size):
            # Skip processing based on mapping strategy
            if batch_mapping == "all-on-first" and b > 0:
                continue
                
            # Get lines for this batch item
            if batch_mapping == "one-to-one" and line_len == batch_size:
                indices = [b]
            elif batch_mapping == "broadcast" or (batch_mapping == "one-to-one" and line_len == 1):
                indices = range(line_len)
            else:  # "all-on-first"
                indices = range(line_len)
                
            # Skip if no lines to draw
            if not indices:
                continue
                
            # Single efficient conversion to numpy
            img_np, device, dtype = self.prep_batch_image(output_batch, b)
            
            # Draw all lines for this image
            for i in indices:
                _x1, _y1 = x1_list[i], y1_list[i]
                _x2, _y2 = x2_list[i], y2_list[i]
                
                if not all(isinstance(c, (float, int)) for c in [_x1, _y1, _x2, _y2]):
                    continue
                    
                # Convert to pixel coordinates if needed
                if is_normalized:
                    px1 = int(max(0, min(width - 1, _x1 * width)))
                    py1 = int(max(0, min(height - 1, _y1 * height)))
                    px2 = int(max(0, min(width - 1, _x2 * width)))
                    py2 = int(max(0, min(height - 1, _y2 * height)))
                else:
                    px1 = int(max(0, min(width - 1, _x1)))
                    py1 = int(max(0, min(height - 1, _y1)))
                    px2 = int(max(0, min(width - 1, _x2)))
                    py2 = int(max(0, min(height - 1, _y2)))
                    
                # Draw line
                cv2.line(img_np, (px1, py1), (px2, py2), bgr_color, thickness)
                
                # Optional features
                if draw_endpoints:
                    cv2.circle(img_np, (px1, py1), point_radius, bgr_color, -1)
                    cv2.circle(img_np, (px2, py2), point_radius, bgr_color, -1)
                    
                if draw_label and label_text:
                    # Calculate label position
                    if label_position == "Midpoint":
                        label_x = int((px1 + px2) / 2)
                        label_y = int((py1 + py2) / 2)
                    elif label_position == "Start":
                        label_x, label_y = px1, py1
                    else:  # End or fallback
                        label_x, label_y = px2, py2
                        
                    # Add offset and draw text
                    label_y += 10
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img_np, label_text, (label_x, label_y), 
                              font, font_scale, bgr_color, 1, cv2.LINE_AA)
                              
            # Convert back to tensor and update batch
            output_batch[b] = self.tensor_from_numpy(img_np, device, dtype)
            
        return output_batch
        
    def draw_polygon(self,
                    image: torch.Tensor,
                    points_x: List[float],
                    points_y: List[float],
                    color: str = "#0000FF",
                    thickness: int = 2,
                    fill: bool = False,
                    is_normalized: bool = True,
                    batch_mapping: str = "broadcast",
                    draw_vertices: bool = False,
                    vertex_radius: int = 3,
                    draw_label: bool = False,
                    label_text: str = "",
                    label_position: str = "Center",
                    font_scale: float = 0.5) -> torch.Tensor:
        """
        Draw polygons on images efficiently.
        
        Args:
            image: BHWC tensor
            points_x/y: Lists of polygon vertex coordinates 
            color: Hex color
            thickness: Line thickness for polygon outline
            fill: Whether to fill the polygon
            is_normalized: If True, coords are 0-1 normalized
            batch_mapping: How to map polygons to batch
            draw_vertices: Whether to draw points at polygon vertices
            vertex_radius: Radius of vertex points
            draw_label: Whether to draw a text label
            label_text: Text to display
            label_position: Position of the label ("Center" only for now)
            font_scale: Font scale
            
        Returns:
            Modified image tensor
        """
        batch_size, height, width, _ = image.shape
        output_batch = image.clone()
        bgr_color = self.hex_to_bgr(color)
        
        # Validate input
        if not isinstance(points_x, list) or not isinstance(points_y, list):
            logger.error("Polygon requires lists of points")
            return output_batch
            
        if len(points_x) != len(points_y):
            logger.error("points_x and points_y must have the same length")
            return output_batch
            
        if len(points_x) < 3:
            logger.error("Polygon requires at least 3 points")
            return output_batch
            
        # Check if we have a list of polygon point sets or just a single polygon
        is_multi_polygon = False
        if isinstance(points_x[0], list):
            is_multi_polygon = True
            if not all(isinstance(p, list) for p in points_x + points_y):
                logger.error("If providing multiple polygons, all entries in points_x/y must be lists")
                return output_batch
            if len(points_x) != len(points_y):
                logger.error("For multiple polygons, points_x and points_y lists must have the same length")
                return output_batch
                
            poly_count = len(points_x)
            # Validate each polygon has matching x,y counts
            for i in range(poly_count):
                if len(points_x[i]) != len(points_y[i]):
                    logger.error(f"Polygon {i}: points_x and points_y must have the same length")
                    return output_batch
                if len(points_x[i]) < 3:
                    logger.error(f"Polygon {i} requires at least 3 points")
                    return output_batch
        else:
            # Single polygon case - wrap in outer list for consistent processing
            points_x = [points_x]
            points_y = [points_y]
            poly_count = 1
            
        # Determine batch mapping
        if batch_mapping == "one-to-one" and poly_count != batch_size and poly_count != 1:
            logger.warning(f"One-to-one mapping requested but polygon count ({poly_count}) != batch size ({batch_size})")
            batch_mapping = "all-on-first"
            
        # Process each image in batch
        for b in range(batch_size):
            # Skip processing based on mapping strategy
            if batch_mapping == "all-on-first" and b > 0:
                continue
                
            # Get polygons for this batch item
            if batch_mapping == "one-to-one" and poly_count == batch_size:
                poly_indices = [b]
            elif batch_mapping == "broadcast" or (batch_mapping == "one-to-one" and poly_count == 1):
                poly_indices = range(poly_count)
            else:  # "all-on-first"
                poly_indices = range(poly_count)
                
            # Skip if no polygons to draw
            if not poly_indices:
                continue
                
            # Single efficient conversion to numpy
            img_np, device, dtype = self.prep_batch_image(output_batch, b)
            
            # Draw all polygons for this image
            for i in poly_indices:
                px = points_x[i]
                py = points_y[i]
                
                # Convert to pixel coordinates if needed
                if is_normalized:
                    # Vectorized conversion
                    px_pixels = [int(max(0, min(width - 1, x * width))) for x in px]
                    py_pixels = [int(max(0, min(height - 1, y * height))) for y in py]
                else:
                    # Ensure within bounds
                    px_pixels = [int(max(0, min(width - 1, x))) for x in px]
                    py_pixels = [int(max(0, min(height - 1, y))) for y in py]
                
                # Combine x,y into points format needed by OpenCV
                points = np.array(list(zip(px_pixels, py_pixels)), np.int32)
                points = points.reshape((-1, 1, 2))
                
                # Draw filled polygon
                if fill:
                    cv2.fillPoly(img_np, [points], bgr_color)
                
                # Draw polygon outline
                cv2.polylines(img_np, [points], True, bgr_color, thickness)
                
                # Draw vertices if requested
                if draw_vertices:
                    for vx, vy in zip(px_pixels, py_pixels):
                        cv2.circle(img_np, (vx, vy), vertex_radius, bgr_color, -1)
                
                # Draw label if requested
                if draw_label and label_text:
                    # Calculate centroid for label position
                    centroid_x = sum(px_pixels) // len(px_pixels)
                    centroid_y = sum(py_pixels) // len(py_pixels)
                    
                    # Add offset and draw text with outline for better visibility
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_size = cv2.getTextSize(label_text, font, font_scale, 1)[0]
                    
                    # Center text
                    text_x = max(0, min(width - text_size[0], centroid_x - text_size[0] // 2))
                    text_y = centroid_y
                    
                    # Draw text outline/shadow for better visibility
                    cv2.putText(img_np, label_text, (text_x, text_y), 
                              font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)  # Thicker black outline
                    # Draw main text
                    cv2.putText(img_np, label_text, (text_x, text_y), 
                              font, font_scale, bgr_color, 1, cv2.LINE_AA)
            
            # Convert back to tensor and update batch
            output_batch[b] = self.tensor_from_numpy(img_np, device, dtype)
            
        return output_batch 