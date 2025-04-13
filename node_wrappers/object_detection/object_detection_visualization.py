# visualize_object_detection.py
import torch
import numpy as np
import cv2

from ...src.utils.image_utils import convert_to_cv2, convert_to_tensor
from ..common.base_visualization_nodes import BaseDetectionVisualizationNode

import logging
logger = logging.getLogger(__name__)
_category = "MediaPipeVision/ObjectDetection/Visualization"

class VisualizeObjectDetections(BaseDetectionVisualizationNode):
    """Draws object detection bounding boxes and labels onto images."""
    CATEGORY = _category

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "object_detections": ("OBJECT_DETECTIONS",), # Expect the specific type
                "image": ("IMAGE",),
            },
            "optional": {
                "box_color": ("STRING", {"default": "#FF0000"}), # Red
                "box_thickness": ("INT", {"default": 2, "min": 1, "max": 10}),
                "label_color": ("STRING", {"default": "#FFFFFF"}), # White
                "label_bg_color": ("STRING", {"default": "#FF0000"}), # Red BG
                # Font size might be tricky to implement well, skipping for now
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_visualizations"

    def draw_visualizations(self, object_detections, image, 
                            box_color="#FF0000", box_thickness=2,
                            label_color="#FFFFFF", label_bg_color="#FF0000"):
        """Main function that processes the batch of images and detections."""
        return self.process_batch(
            object_detections, 
            image,
            self.draw_objects,
            box_color=box_color,
            box_thickness=box_thickness,
            label_color=label_color,
            label_bg_color=label_bg_color
        )
    
    def draw_objects(self, image, width, height, detections, 
                     box_color="#FF0000", box_thickness=2,
                     label_color="#FFFFFF", label_bg_color="#FF0000"):
        """Callback function to draw objects on a single image."""
        # Convert hex colors to BGR tuples for cv2
        box_color_bgr = self._hex_to_bgr(box_color)
        
        for detection in detections:
            bbox = detection.bounding_box
            # Top-left corner
            x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
            # Bottom-right corner
            x2, y2 = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color_bgr, box_thickness)
            
            # Prepare label text (use first category's display name if available)
            label = "N/A"
            score = 0.0
            if detection.categories:
                top_category = detection.categories[0]
                label = top_category.display_name or top_category.category_name or "N/A"
                score = top_category.score
            
            label_text = f"{label}: {score:.2f}"
            
            # Draw label background and text inside the box top-left
            try:
                self.draw_text_with_background(
                    image, 
                    label_text, 
                    (x1, y1),
                    text_color=label_color,
                    bg_color=label_bg_color
                )
            except Exception as e:
                logger.error(f"Error drawing text label: {e}")
                # Fallback: Draw text without background
                cv2.putText(
                    image, 
                    label_text, 
                    (x1 + 2, y1 + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    self._hex_to_bgr(label_color), 
                    1
                )

# Node registration
NODE_CLASS_MAPPINGS = {
    "VisualizeObjectDetections": VisualizeObjectDetections
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisualizeObjectDetections": "Visualize Object Detections (MediaPipe)"
} 