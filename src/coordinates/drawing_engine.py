import logging
from typing import List, Tuple, Union
import math

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
    def hex_to_rgb_tensor(hex_color: str, device: torch.device) -> torch.Tensor:
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            return torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
        try:
            rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            return torch.tensor([rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0], device=device, dtype=torch.float32)
        except ValueError:
            return torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)

    @staticmethod
    def create_coordinate_grids(height: int, width: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        y_coords = torch.arange(height, device=device, dtype=torch.float32).view(-1, 1).expand(height, width)
        x_coords = torch.arange(width, device=device, dtype=torch.float32).view(1, -1).expand(height, width)
        return y_coords, x_coords

    @staticmethod
    def draw_circles_gpu(image: torch.Tensor, coords: List[Tuple[int, int]], radius: int, color: torch.Tensor) -> torch.Tensor:
        if not coords:
            return image
        
        batch_size, height, width, channels = image.shape
        device = image.device
        
        # Create coordinate grids once
        y_grid, x_grid = DrawingEngine.create_coordinate_grids(height, width, device)
        
        for px, py in coords:
            # Calculate squared distances (avoid sqrt for performance)
            dist_sq = (x_grid - px) ** 2 + (y_grid - py) ** 2
            mask = dist_sq <= radius ** 2
            
            # Apply color to all batch items and channels
            for b in range(batch_size):
                for c in range(channels):
                    image[b, :, :, c][mask] = color[c]
        
        return image

    @staticmethod
    def draw_lines_gpu(image: torch.Tensor, lines: List[Tuple[int, int, int, int]], thickness: int, color: torch.Tensor) -> torch.Tensor:
        if not lines:
            return image
            
        batch_size, height, width, channels = image.shape
        device = image.device
        
        # Create coordinate grids once
        y_grid, x_grid = DrawingEngine.create_coordinate_grids(height, width, device)
        
        half_thickness = thickness / 2.0
        
        for px1, py1, px2, py2 in lines:
            # Handle degenerate cases
            if px1 == px2 and py1 == py2:
                # Single point
                mask = (x_grid - px1) ** 2 + (y_grid - py1) ** 2 <= half_thickness ** 2
            else:
                # Line segment using distance to line formula
                # Vector from point1 to point2
                dx = px2 - px1
                dy = py2 - py1
                line_length_sq = dx * dx + dy * dy
                
                if line_length_sq == 0:
                    continue
                
                # Vector from point1 to each pixel
                px_diff = x_grid - px1
                py_diff = y_grid - py1
                
                # Project onto line segment (0 <= t <= 1)
                t = torch.clamp((px_diff * dx + py_diff * dy) / line_length_sq, 0, 1)
                
                # Closest point on line segment
                closest_x = px1 + t * dx
                closest_y = py1 + t * dy
                
                # Distance from pixel to closest point on line
                dist_sq = (x_grid - closest_x) ** 2 + (y_grid - closest_y) ** 2
                mask = dist_sq <= half_thickness ** 2
            
            # Apply color to all batch items and channels
            for b in range(batch_size):
                for c in range(channels):
                    image[b, :, :, c][mask] = color[c]
        
        return image

    @staticmethod
    def draw_polygon_gpu(image: torch.Tensor, points: List[Tuple[int, int]], fill: bool, thickness: int, color: torch.Tensor) -> torch.Tensor:
        if len(points) < 3:
            return image
            
        batch_size, height, width, channels = image.shape
        device = image.device
        
        if fill:
            # Vectorized point-in-polygon test using GPU
            y_grid, x_grid = DrawingEngine.create_coordinate_grids(height, width, device)
            
            # Convert points to tensors
            points_tensor = torch.tensor(points, device=device, dtype=torch.float32)
            x_coords = points_tensor[:, 0]
            y_coords = points_tensor[:, 1]
            
            # Initialize mask
            mask = torch.zeros(height, width, device=device, dtype=torch.bool)
            
            # Vectorized ray casting algorithm
            n_points = len(points)
            for i in range(n_points):
                j = (i - 1) % n_points
                
                xi, yi = x_coords[i], y_coords[i]
                xj, yj = x_coords[j], y_coords[j]
                
                # Check if ray crosses edge
                cond1 = (yi > y_grid) != (yj > y_grid)
                
                # Calculate intersection x-coordinate
                denom = yj - yi
                # Avoid division by zero
                valid_denom = torch.abs(denom) > 1e-10
                intersection_x = torch.where(
                    valid_denom,
                    (xj - xi) * (y_grid - yi) / denom + xi,
                    torch.full_like(x_grid, float('inf'))
                )
                
                cond2 = x_grid < intersection_x
                
                # Toggle mask where both conditions are true
                edge_cross = cond1 & cond2 & valid_denom
                mask = mask ^ edge_cross
            
            # Apply color to filled area
            for b in range(batch_size):
                for c in range(channels):
                    image[b, :, :, c][mask] = color[c]
        else:
            # Draw outline by connecting line segments
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                image = DrawingEngine.draw_lines_gpu(image, [(*p1, *p2)], thickness, color)
        
        return image

    @staticmethod
    def prep_batch_image(image: torch.Tensor, batch_idx: int) -> Tuple[np.ndarray, torch.device, torch.dtype]:
        """Fast preparation of single image from batch for drawing - minimal conversion."""
        if image.dim() != 4:
            raise ValueError("Input image must be BHWC format")

        device, dtype = image.device, image.dtype
        img_tensor = image[batch_idx].to(torch.float32)
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        return np.ascontiguousarray(img_np), device, dtype

    @staticmethod
    def tensor_from_numpy(img_np: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Fast conversion from numpy back to tensor with proper device/dtype."""
        return torch.from_numpy(img_np.astype(np.float32) / 255.0).to(device=device, dtype=dtype)

    @staticmethod
    def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            return (0, 0, 0)
        try:
            rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            return (rgb[2], rgb[1], rgb[0])
        except ValueError:
            return (0, 0, 0)

    def draw_points(
        self,
        image: torch.Tensor,
        points_x: Union[float, List[float]],
        points_y: Union[float, List[float]],
        radius: int = 5,
        color: str = "#FF0000",
        is_normalized: bool = True,
        batch_mapping: str = "broadcast",
    ) -> torch.Tensor:
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
        device = image.device
        color_tensor = self.hex_to_rgb_tensor(color, device)

        x_list = points_x if isinstance(points_x, list) else [points_x]
        y_list = points_y if isinstance(points_y, list) else [points_y]

        points_len = len(x_list)
        if batch_mapping == "one-to-one" and points_len != batch_size and points_len != 1:
            logger.warning(f"One-to-one mapping requested but point list length ({points_len}) != batch size ({batch_size})")
            batch_mapping = "all-on-first"

        # Pre-filter and convert coordinates
        valid_coords = []
        for xi, yi in zip(x_list, y_list):
            if isinstance(xi, (float, int)) and isinstance(yi, (float, int)):
                if is_normalized:
                    px = int(max(0, min(width - 1, CoordinateSystem.denormalize(xi, width, CoordinateSystem.PIXEL))))
                    py = int(max(0, min(height - 1, CoordinateSystem.denormalize(yi, height, CoordinateSystem.PIXEL))))
                else:
                    px = int(max(0, min(width - 1, xi)))
                    py = int(max(0, min(height - 1, yi)))
                valid_coords.append((px, py))

        if not valid_coords:
            return image

        # Process each batch item
        for b in range(batch_size):
            if batch_mapping == "all-on-first" and b > 0:
                continue

            if batch_mapping == "one-to-one" and points_len == batch_size:
                if b < len(valid_coords):
                    coords_to_draw = [valid_coords[b]]
                else:
                    continue
            else:
                coords_to_draw = valid_coords

            if coords_to_draw:
                # Extract single batch item for processing
                batch_image = image[b:b+1]
                batch_image = self.draw_circles_gpu(batch_image, coords_to_draw, radius, color_tensor)
                image[b:b+1] = batch_image

        return image

    def draw_lines(
        self,
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
        font_scale: float = 0.5,
    ) -> torch.Tensor:
        batch_size, height, width, _ = image.shape
        device = image.device
        color_tensor = self.hex_to_rgb_tensor(color, device)

        x1_list = x1 if isinstance(x1, list) else [x1]
        y1_list = y1 if isinstance(y1, list) else [y1]
        x2_list = x2 if isinstance(x2, list) else [x2]
        y2_list = y2 if isinstance(y2, list) else [y2]

        line_len = len(x1_list)
        if not all(len(coord) == line_len for coord in [y1_list, x2_list, y2_list]):
            raise ValueError("All coordinate lists must have the same length")

        if batch_mapping == "one-to-one" and line_len != batch_size and line_len != 1:
            logger.warning(f"One-to-one mapping requested but line list length ({line_len}) != batch size ({batch_size})")
            batch_mapping = "all-on-first"

        # Pre-filter and convert coordinates
        valid_lines = []
        endpoint_coords = []
        for i in range(line_len):
            _x1, _y1, _x2, _y2 = x1_list[i], y1_list[i], x2_list[i], y2_list[i]
            if all(isinstance(c, (float, int)) for c in [_x1, _y1, _x2, _y2]):
                if is_normalized:
                    px1 = int(max(0, min(width - 1, CoordinateSystem.denormalize(_x1, width, CoordinateSystem.PIXEL))))
                    py1 = int(max(0, min(height - 1, CoordinateSystem.denormalize(_y1, height, CoordinateSystem.PIXEL))))
                    px2 = int(max(0, min(width - 1, CoordinateSystem.denormalize(_x2, width, CoordinateSystem.PIXEL))))
                    py2 = int(max(0, min(height - 1, CoordinateSystem.denormalize(_y2, height, CoordinateSystem.PIXEL))))
                else:
                    px1 = int(max(0, min(width - 1, _x1)))
                    py1 = int(max(0, min(height - 1, _y1)))
                    px2 = int(max(0, min(width - 1, _x2)))
                    py2 = int(max(0, min(height - 1, _y2)))
                valid_lines.append((px1, py1, px2, py2))
                if draw_endpoints:
                    endpoint_coords.extend([(px1, py1), (px2, py2)])

        if not valid_lines:
            return image

        # Process each batch item
        for b in range(batch_size):
            if batch_mapping == "all-on-first" and b > 0:
                continue

            if batch_mapping == "one-to-one" and line_len == batch_size:
                if b < len(valid_lines):
                    lines_to_draw = [valid_lines[b]]
                    endpoints_to_draw = []
                    if draw_endpoints:
                        line_data = valid_lines[b]
                        endpoints_to_draw = [(line_data[0], line_data[1]), (line_data[2], line_data[3])]
                else:
                    continue
            else:
                lines_to_draw = valid_lines
                endpoints_to_draw = endpoint_coords

            if lines_to_draw:
                # Extract single batch item for processing
                batch_image = image[b:b+1]
                
                # Draw lines using GPU
                batch_image = self.draw_lines_gpu(batch_image, lines_to_draw, thickness, color_tensor)
                
                # Draw endpoints using GPU
                if endpoints_to_draw:
                    batch_image = self.draw_circles_gpu(batch_image, endpoints_to_draw, point_radius, color_tensor)
                
                # Handle text labels (still use CV2 for now)
                if draw_label and label_text:
                    # Convert to numpy only for text rendering
                    img_np, device_orig, dtype_orig = self.prep_batch_image(batch_image, 0)
                    bgr_color = self.hex_to_bgr(color)
                    
                    for line_data in lines_to_draw:
                        px1, py1, px2, py2 = line_data
                        
                        if label_position == "Midpoint":
                            label_x = int((px1 + px2) / 2)
                            label_y = int((py1 + py2) / 2)
                        elif label_position == "Start":
                            label_x, label_y = px1, py1
                        else:
                            label_x, label_y = px2, py2

                        text_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

                        cv2.rectangle(
                            img_np,
                            (label_x - 5, label_y - text_size[1] - 5),
                            (label_x + text_size[0] + 5, label_y + 5),
                            (50, 50, 50),
                            -1,
                        )

                        cv2.putText(img_np, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
                    
                    batch_image[0] = self.tensor_from_numpy(img_np, device_orig, dtype_orig)
                
                image[b:b+1] = batch_image

        return image

    def draw_polygon(
        self,
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
        font_scale: float = 0.5,
    ) -> torch.Tensor:
        batch_size, height, width, _ = image.shape
        device = image.device
        color_tensor = self.hex_to_rgb_tensor(color, device)

        if len(points_x) != len(points_y):
            raise ValueError("X and Y coordinate lists must have the same length")

        if len(points_x) < 3:
            logger.warning("Polygon needs at least 3 points. Drawing nothing.")
            return image

        if batch_mapping == "one-to-one" and batch_size > 1:
            logger.warning(f"One-to-one mapping not supported for polygons. Using broadcast.")
            batch_mapping = "broadcast"

        # Pre-filter and convert coordinates
        pixel_points = []
        for i in range(len(points_x)):
            x_coord = points_x[i]
            y_coord = points_y[i]

            if isinstance(x_coord, (float, int)) and isinstance(y_coord, (float, int)):
                if is_normalized:
                    px = int(max(0, min(width - 1, CoordinateSystem.denormalize(x_coord, width, CoordinateSystem.PIXEL))))
                    py = int(max(0, min(height - 1, CoordinateSystem.denormalize(y_coord, height, CoordinateSystem.PIXEL))))
                else:
                    px = int(max(0, min(width - 1, x_coord)))
                    py = int(max(0, min(height - 1, y_coord)))
                pixel_points.append((px, py))

        if len(pixel_points) < 3:
            return image

        # Process each batch item
        for b in range(batch_size):
            if batch_mapping == "all-on-first" and b > 0:
                continue

            # Extract single batch item for processing
            batch_image = image[b:b+1]
            
            # Draw polygon using GPU
            batch_image = self.draw_polygon_gpu(batch_image, pixel_points, fill, thickness, color_tensor)
            
            # Draw vertices using GPU if requested
            if draw_vertices:
                batch_image = self.draw_circles_gpu(batch_image, pixel_points, vertex_radius, color_tensor)

            # Handle text labels (still use CV2 for now)
            if draw_label and label_text:
                # Convert to numpy only for text rendering
                img_np, device_orig, dtype_orig = self.prep_batch_image(batch_image, 0)
                bgr_color = self.hex_to_bgr(color)
                
                center_x = int(sum(p[0] for p in pixel_points) / len(pixel_points))
                center_y = int(sum(p[1] for p in pixel_points) / len(pixel_points))

                text_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

                cv2.rectangle(
                    img_np,
                    (center_x - text_size[0] // 2 - 5, center_y - text_size[1] // 2 - 5),
                    (center_x + text_size[0] // 2 + 5, center_y + text_size[1] // 2 + 5),
                    (50, 50, 50),
                    -1,
                )

                cv2.putText(
                    img_np,
                    label_text,
                    (center_x - text_size[0] // 2, center_y + text_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                )
                
                batch_image[0] = self.tensor_from_numpy(img_np, device_orig, dtype_orig)

            image[b:b+1] = batch_image

        return image
