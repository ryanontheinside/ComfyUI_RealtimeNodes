# visualize_face_detection.py
import torch
import numpy as np
import cv2
from ..common.base_visualization_nodes import BaseDetectionVisualizationNode

import logging
logger = logging.getLogger(__name__)

class VisualizeFaceDetections(BaseDetectionVisualizationNode):
    """Draws face detection bounding boxes and keypoints onto images."""
    CATEGORY = "MediaPipeVision/Face/FaceDetection/Visualization"
    DESCRIPTION = "Visualizes face detections by drawing boxes and keypoints onto images. Allows customization of colors and sizes to highlight detected faces."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_detections": ("FACE_DETECTIONS", {"tooltip": "Face detection results from the Face Detector node"}),
                "image": ("IMAGE", {"tooltip": "The image on which to draw face detection visualizations"}),
            },
            "optional": {
                "box_color": ("STRING", {"default": "#00FF00", "tooltip": "Hex color code for the face bounding boxes (default: green)"}),
                "box_thickness": ("INT", {"default": 2, "min": 1, "max": 10, "tooltip": "Thickness of the face bounding box lines"}),
                "keypoint_color": ("STRING", {"default": "#FF0000", "tooltip": "Hex color code for facial keypoints (default: red)"}),
                "keypoint_radius": ("INT", {"default": 2, "min": 1, "max": 10, "tooltip": "Size of the dots representing facial keypoints"}),
                "label_color": ("STRING", {"default": "#FFFFFF", "tooltip": "Hex color code for the text labels (default: white)"}),
                "label_bg_color": ("STRING", {"default": "#00FF00", "tooltip": "Hex color code for the label background (default: green)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_visualizations"

    def draw_visualizations(self, face_detections, image, 
                            box_color="#00FF00", box_thickness=2,
                            keypoint_color="#FF0000", keypoint_radius=2,
                            label_color="#FFFFFF", label_bg_color="#00FF00"):
        """Main function that processes the batch of images and detections."""
        return self.process_batch(
            face_detections, 
            image,
            self.draw_faces,
            box_color=box_color,
            box_thickness=box_thickness,
            keypoint_color=keypoint_color,
            keypoint_radius=keypoint_radius,
            label_color=label_color,
            label_bg_color=label_bg_color
        )
    
    def draw_faces(self, image, width, height, detections,
                   box_color="#00FF00", box_thickness=2,
                   keypoint_color="#FF0000", keypoint_radius=2,
                   label_color="#FFFFFF", label_bg_color="#00FF00"):
        """Callback function to draw faces on a single image."""
        # Convert hex colors to BGR tuples for cv2
        box_color_bgr = self._hex_to_bgr(box_color)
        keypoint_color_bgr = self._hex_to_bgr(keypoint_color)
        
        for detection in detections:
            bbox = detection.bounding_box
            x1, y1 = bbox.origin_x, bbox.origin_y
            x2, y2 = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            
            # Draw bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), box_color_bgr, box_thickness)
            
            # Draw Keypoints
            if detection.keypoints:
                for kp in detection.keypoints:
                    cx, cy = int(kp.x * width), int(kp.y * height)
                    cv2.circle(image, (cx, cy), keypoint_radius, keypoint_color_bgr, -1)

            # Prepare label text (confidence score)
            label_text = f"Face: {detection.score:.2f}" if detection.score is not None else "Face"
            
            # Draw label with background
            try:
                self.draw_text_with_background(
                    image, 
                    label_text, 
                    (int(x1), int(y1)),
                    text_color=label_color,
                    bg_color=label_bg_color
                )
            except Exception as e:
                logger.error(f"Error drawing text label: {e}")
                # Fallback using basic cv2 text
                cv2.putText(
                    image, 
                    label_text, 
                    (int(x1) + 2, int(y1) + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    self._hex_to_bgr(label_color), 
                    1
                )

# Node registration
NODE_CLASS_MAPPINGS = {
    "VisualizeFaceDetections": VisualizeFaceDetections
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisualizeFaceDetections": "Visualize Face Detections (MediaPipe)"
} 