import logging
from typing import List, Dict, Tuple, Any, Optional, Type
import cv2
import numpy as np

from ...src.utils.image_utils import convert_to_cv2, convert_to_tensor

logger = logging.getLogger(__name__)

# Default font setup - can be used by all visualization classes
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_FONT_SCALE = 0.5
DEFAULT_FONT_THICKNESS = 1

class BaseLandmarkVisualizationNode:
    """Base class for landmark visualization nodes."""
    CATEGORY = "MediaPipeVision/Visualization"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_visualizations"
    
    # --- To be overridden by subclasses --- 
    LANDMARKS_TYPE = "ANY" # The specific ComfyUI landmark type string (e.g., "FACE_LANDMARKS")
    CONNECTIONS = None     # MediaPipe connections list (e.g., mp.solutions.face_mesh.FACEMESH_TESSELATION)
    DEFAULT_DRAW_POINTS = True
    DEFAULT_POINT_COLOR = "#FFFFFF"
    DEFAULT_POINT_RADIUS = 3
    DEFAULT_DRAW_CONNECTIONS = True
    DEFAULT_CONNECTION_COLOR = "#00FF00"
    DEFAULT_CONNECTION_THICKNESS = 2
    SUPPORTS_VISIBILITY = False # Does this landmark type provide visibility scores?
    # --- End Override Section --- 
    
    @classmethod
    def INPUT_TYPES(cls):
        """Generate input types based on class properties."""
        inputs = {
            "required": {
                "landmarks": (cls.LANDMARKS_TYPE,),
                "image": ("IMAGE",),
            },
            "optional": {
                "draw_points": ("BOOLEAN", {"default": cls.DEFAULT_DRAW_POINTS}),
                "point_color": ("STRING", {"default": cls.DEFAULT_POINT_COLOR}),
                "point_radius": ("INT", {"default": cls.DEFAULT_POINT_RADIUS, "min": 1, "max": 10}),
                "draw_connections": ("BOOLEAN", {"default": cls.DEFAULT_DRAW_CONNECTIONS}),
                "connection_color": ("STRING", {"default": cls.DEFAULT_CONNECTION_COLOR}),
                "connection_thickness": ("INT", {"default": cls.DEFAULT_CONNECTION_THICKNESS, "min": 1, "max": 10}),
            }
        }
        
        if cls.SUPPORTS_VISIBILITY:
            inputs["optional"]["min_visibility"] = ("FLOAT", {
                "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Minimum visibility score to draw a point/connection endpoint"
            })
            
        return inputs
    
    def draw_visualizations(self, landmarks, image, **kwargs):
        """Main execution function: handles batching, image conversion, and calls drawing logic."""
        
        # Extract relevant optional arguments using class defaults
        draw_points = kwargs.get('draw_points', self.DEFAULT_DRAW_POINTS)
        point_color = kwargs.get('point_color', self.DEFAULT_POINT_COLOR)
        point_radius = kwargs.get('point_radius', self.DEFAULT_POINT_RADIUS)
        draw_connections = kwargs.get('draw_connections', self.DEFAULT_DRAW_CONNECTIONS)
        connection_color = kwargs.get('connection_color', self.DEFAULT_CONNECTION_COLOR)
        connection_thickness = kwargs.get('connection_thickness', self.DEFAULT_CONNECTION_THICKNESS)
        min_visibility = kwargs.get('min_visibility', 0.5) # Default if not supported/provided
        
        if not landmarks:
            logger.warning(f"{self.__class__.__name__}: No landmarks provided.")
            return (image,)

        cv2_images = convert_to_cv2(image)
        processed_cv2_images = []
        batch_size = len(cv2_images)

        if len(landmarks) != batch_size:
            logger.error(f"{self.__class__.__name__}: Batch size mismatch. Images: {batch_size}, Landmark sets: {len(landmarks)}. Cannot proceed reliably.")
            return (image,)

        # Convert hex colors to BGR tuples
        point_color_bgr = self._hex_to_bgr(point_color)
        connection_color_bgr = self._hex_to_bgr(connection_color)

        for i, cv2_image in enumerate(cv2_images):
            height, width = cv2_image.shape[:2]
            landmarks_for_image = landmarks[i]  # List of landmark results for this image item

            if not landmarks_for_image:
                processed_cv2_images.append(cv2_image) # Append unmodified image
                continue

            # Iterate through all detections (faces, hands, poses) in this image item
            for single_detection_result in landmarks_for_image:
                self.draw_single_detection(cv2_image, width, height, single_detection_result,
                                           draw_points, point_color_bgr, point_radius,
                                           draw_connections, connection_color_bgr, connection_thickness,
                                           min_visibility)
            
            processed_cv2_images.append(cv2_image)

        output_image = convert_to_tensor(processed_cv2_images)
        return (output_image,)
    
    def _hex_to_bgr(self, hex_color):
        """Convert hex color string to BGR tuple."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)  # OpenCV uses BGR
    
    def draw_single_detection(self, cv2_image, width: int, height: int, 
                              detection_result: Any,
                              draw_points: bool, point_color, point_radius: int,
                              draw_connections: bool, connection_color, connection_thickness: int,
                              min_visibility: float):
        """Draws landmarks and connections for a single detected object (face, hand, pose)."""
        
        # Extract the actual list of landmarks (this might vary slightly based on MediaPipe result structure)
        landmark_list = self.get_landmark_list(detection_result)
        if not landmark_list:
            return # Skip if no landmarks found for this detection
            
        points = {}
        # Process landmarks: calculate coordinates and optionally draw points
        for lm in landmark_list:
            # Check visibility if supported
            is_visible = True
            if self.SUPPORTS_VISIBILITY:
                 vis = lm.visibility if hasattr(lm, 'visibility') and lm.visibility is not None else 1.0
                 is_visible = vis >= min_visibility
                 
            if is_visible:
                cx, cy = int(lm.x * width), int(lm.y * height)
                points[lm.index] = (cx, cy) # Store denormalized point coordinates
                
                if draw_points:
                    cv2.circle(cv2_image, (cx, cy), point_radius, point_color, -1)
        
        # Draw connections if enabled and applicable
        if draw_connections and self.CONNECTIONS and points:
            for connection in self.CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                # Draw connection only if both points are present (i.e., passed visibility check)
                if start_idx in points and end_idx in points:
                    cv2.line(cv2_image, points[start_idx], points[end_idx], connection_color, connection_thickness)

    def get_landmark_list(self, detection_result: Any) -> Optional[List[Any]]:
        """Abstract method to extract the list of landmark objects from a single detection result.
           Needs to be implemented by subclasses based on the specific MediaPipe result structure.
           e.g., for hands/pose: return detection_result.landmarks
                 for face: return detection_result (it's already the list)
        """
        raise NotImplementedError("Subclasses must implement get_landmark_list")


class BaseDetectionVisualizationNode:
    """Base class for visualization nodes that need to draw bounding boxes and labels.
    
    This can be used for object detection, face detection, and similar tasks
    that require drawing boxes and text labels with a consistent font rendering approach.
    """
    CATEGORY = "MediaPipeVision/Visualization"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_visualizations"
    
    # Default font settings
    FONT = DEFAULT_FONT
    FONT_SCALE = DEFAULT_FONT_SCALE
    FONT_THICKNESS = DEFAULT_FONT_THICKNESS
    
    @classmethod
    def get_text_dimensions(cls, text, font_scale=None, thickness=None):
        """Gets text dimensions using cv2.
        
        Args:
            text: The text to measure
            font_scale: The font scale (defaults to cls.FONT_SCALE)
            thickness: The font thickness (defaults to cls.FONT_THICKNESS)
        
        Returns:
            Tuple of (width, height) in pixels
        """
        if font_scale is None:
            font_scale = cls.FONT_SCALE
        if thickness is None:
            thickness = cls.FONT_THICKNESS
            
        try:
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cls.FONT, font_scale, thickness
            )
            return text_width, text_height + baseline
        except Exception as e:
            logger.error(f"Error measuring text dimensions: {e}")
            return 100, 20  # Default fallback
    
    def _hex_to_bgr(self, hex_color):
        """Convert hex color string to BGR tuple."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)  # OpenCV uses BGR
    
    def draw_text_with_background(self, image, text, position, 
                                 text_color="#FFFFFF", bg_color="#000000", 
                                 padding=(4, 2), font_scale=None, thickness=None):
        """Draws text with a background rectangle.
        
        Args:
            image: cv2 image to draw on
            text: Text to draw
            position: (x, y) position for top-left of text
            text_color: Hex color for text
            bg_color: Hex color for background
            padding: (x_pad, y_pad) padding around text
            font_scale: Font scale (defaults to cls.FONT_SCALE)
            thickness: Font thickness (defaults to cls.FONT_THICKNESS)
        """
        if font_scale is None:
            font_scale = self.FONT_SCALE
        if thickness is None:
            thickness = self.FONT_THICKNESS
            
        # Convert hex colors to BGR tuples
        text_color_bgr = self._hex_to_bgr(text_color)
        bg_color_bgr = self._hex_to_bgr(bg_color)
            
        x, y = position
        text_width, text_height = self.get_text_dimensions(text, font_scale, thickness)
        
        # Draw background rectangle
        x_pad, y_pad = padding
        cv2.rectangle(
            image,
            (x, y),
            (x + text_width + x_pad*2, y + text_height + y_pad*2),
            bg_color_bgr,
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            image,
            text,
            (x + x_pad, y + text_height),
            self.FONT,
            font_scale,
            text_color_bgr,
            thickness
        )
        
        return text_width, text_height
    
    def process_batch(self, detections, image, drawing_function, **kwargs):
        """Process a batch of images and detections.
        
        Args:
            detections: List of detection results (one per image)
            image: Batch of images as tensor
            drawing_function: Function to call for each image
            **kwargs: Additional arguments for the drawing function
            
        Returns:
            Tuple with processed image tensor
        """
        if not detections:
            logger.warning(f"{self.__class__.__name__}: No detections provided.")
            return (image,)

        cv2_images = convert_to_cv2(image)
        processed_cv2_images = []
        batch_size = len(cv2_images)

        if len(detections) != batch_size:
            logger.error(f"{self.__class__.__name__}: Batch size mismatch. Images: {batch_size}, Detection sets: {len(detections)}. Cannot proceed.")
            return (image,)

        for i, cv2_image in enumerate(cv2_images):
            height, width = cv2_image.shape[:2]
            detections_for_image = detections[i]

            if not detections_for_image:
                processed_cv2_images.append(cv2_image)
                continue

            # Call the drawing function with the specific parameters
            drawing_function(cv2_image, width, height, detections_for_image, **kwargs)
            processed_cv2_images.append(cv2_image)

        output_image = convert_to_tensor(processed_cv2_images)
        return (output_image,) 