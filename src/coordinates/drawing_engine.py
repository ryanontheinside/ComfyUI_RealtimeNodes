import logging
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch

from .coordinate_system import CoordinateSystem

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
                    px = int(CoordinateSystem.denormalize(xi, width, CoordinateSystem.PIXEL))
                    py = int(CoordinateSystem.denormalize(yi, height, CoordinateSystem.PIXEL))
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
                    px1 = int(CoordinateSystem.denormalize(_x1, width, CoordinateSystem.PIXEL))
                    py1 = int(CoordinateSystem.denormalize(_y1, height, CoordinateSystem.PIXEL))
                    px2 = int(CoordinateSystem.denormalize(_x2, width, CoordinateSystem.PIXEL))
                    py2 = int(CoordinateSystem.denormalize(_y2, height, CoordinateSystem.PIXEL))
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
                    else:  # "End"
                        label_x, label_y = px2, py2
                        
                    # Get text size
                    text_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                    
                    # Draw background rectangle
                    cv2.rectangle(img_np, 
                                 (label_x - 5, label_y - text_size[1] - 5),
                                 (label_x + text_size[0] + 5, label_y + 5),
                                 (50, 50, 50), -1)
                    
                    # Draw text
                    cv2.putText(img_np, label_text, (label_x, label_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            
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
        Draw polygon on images efficiently.
        
        Args:
            image: BHWC tensor
            points_x/y: Lists of polygon vertex coordinates
            thickness: Line thickness in pixels
            color: Hex color
            fill: Whether to fill the polygon
            is_normalized: If True, coords are 0-1 normalized
            batch_mapping: How to map polygons to batch
            Other params: Styling options
            
        Returns:
            Modified image tensor
        """
        batch_size, height, width, _ = image.shape
        output_batch = image.clone()
        bgr_color = self.hex_to_bgr(color)
        
        # Validate input lengths
        if len(points_x) != len(points_y):
            raise ValueError("X and Y coordinate lists must have the same length")
            
        # Ensure lists have at least 3 points for a polygon
        if len(points_x) < 3:
            logger.warning("Polygon needs at least 3 points. Drawing nothing.")
            return output_batch
            
        # Determine batch mapping
        if batch_mapping == "one-to-one" and batch_size > 1:
            logger.warning(f"One-to-one mapping not supported for polygons. Using broadcast.")
            batch_mapping = "broadcast"
            
        # Process each image in batch
        for b in range(batch_size):
            # Skip processing based on mapping strategy
            if batch_mapping == "all-on-first" and b > 0:
                continue
                
            # Single efficient conversion to numpy
            img_np, device, dtype = self.prep_batch_image(output_batch, b)
            
            # Convert polygon points to pixel coordinates
            pixel_points = []
            for i in range(len(points_x)):
                x_coord = points_x[i]
                y_coord = points_y[i]
                
                if not isinstance(x_coord, (float, int)) or not isinstance(y_coord, (float, int)):
                    continue
                    
                # Convert to pixel coordinates if needed
                if is_normalized:
                    px = int(CoordinateSystem.denormalize(x_coord, width, CoordinateSystem.PIXEL))
                    py = int(CoordinateSystem.denormalize(y_coord, height, CoordinateSystem.PIXEL))
                else:
                    px = int(max(0, min(width - 1, x_coord)))
                    py = int(max(0, min(height - 1, y_coord)))
                    
                pixel_points.append((px, py))
                
                # Draw vertices if requested
                if draw_vertices:
                    cv2.circle(img_np, (px, py), vertex_radius, bgr_color, -1)
            
            # Skip if no valid points
            if len(pixel_points) < 3:
                continue
                
            # Convert to numpy array for OpenCV
            pts = np.array(pixel_points, dtype=np.int32)
            
            # Draw filled or outline polygon
            if fill:
                cv2.fillPoly(img_np, [pts], bgr_color)
            else:
                cv2.polylines(img_np, [pts], True, bgr_color, thickness)
                
            # Draw label if requested
            if draw_label and label_text:
                # Calculate label position (center of polygon)
                center_x = int(sum(p[0] for p in pixel_points) / len(pixel_points))
                center_y = int(sum(p[1] for p in pixel_points) / len(pixel_points))
                
                # Get text size
                text_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                
                # Draw background rectangle
                cv2.rectangle(img_np, 
                             (center_x - text_size[0]//2 - 5, center_y - text_size[1]//2 - 5),
                             (center_x + text_size[0]//2 + 5, center_y + text_size[1]//2 + 5),
                             (50, 50, 50), -1)
                
                # Draw text
                cv2.putText(img_np, label_text, 
                           (center_x - text_size[0]//2, center_y + text_size[1]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            
            # Convert back to tensor and update batch
            output_batch[b] = self.tensor_from_numpy(img_np, device, dtype)
            
        return output_batch 