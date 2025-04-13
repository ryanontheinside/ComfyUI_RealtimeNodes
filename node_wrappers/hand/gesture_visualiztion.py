# visualize_gestures.py
import torch
import numpy as np
import cv2

from ...src.utils.image_utils import convert_to_cv2, convert_to_tensor
from ..common.base_visualization_nodes import BaseDetectionVisualizationNode

import logging
logger = logging.getLogger(__name__)

_category = "MediaPipeVision/Hand/GestureVisualization"

class VisualizeGestureRecognitions(BaseDetectionVisualizationNode):
    """Draws recognized gesture labels onto images near the detected hand."""
    CATEGORY = _category
    DESCRIPTION = "Visualizes recognized hand gestures by drawing text labels next to hands in the image. Displays gesture names like 'Thumbs Up', 'Victory', 'Open Palm' etc. near each detected hand."
    
    # Override font scale for gesture labels (larger than default)
    FONT_SCALE = 0.7  # Larger than default for gesture labels

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gesture_recognitions": ("GESTURE_RECOGNITIONS", {"tooltip": "Gesture recognition results from the Gesture Recognizer node"}),
                "hand_landmarks": ("HAND_LANDMARKS", {"tooltip": "Hand landmark data from the Hand Landmarker node - needed to position the text labels"}),
                "image": ("IMAGE", {"tooltip": "The image on which to draw gesture labels"}),
            },
            "optional": {
                "label_color": ("STRING", {"default": "#FFFFFF", "tooltip": "Hex color code for the gesture label text (default: white)"}),
                "label_bg_color": ("STRING", {"default": "#0000FF", "tooltip": "Hex color code for the label background (default: blue)"}),
                "y_offset": ("INT", {"default": -30, "step": 5, "tooltip": "Vertical position adjustment for text labels relative to the wrist - negative moves up, positive moves down"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_visualizations"

    def draw_visualizations(self, gesture_recognitions, hand_landmarks, image, 
                            label_color="#FFFFFF", label_bg_color="#0000FF", y_offset=-30):
        """Main validation and batch processing function."""
        if not gesture_recognitions:
            logger.warning("VisualizeGestureRecognitions: No gestures provided.")
            return (image,)
        if not hand_landmarks:
             logger.warning("VisualizeGestureRecognitions: Hand landmarks required for positioning text.")
             return (image,)
             
        # Basic batch size check
        batch_size = len(image)
        if len(gesture_recognitions) != batch_size or len(hand_landmarks) != batch_size:
            logger.error(f"VisualizeGestureRecognitions: Batch size mismatch. Images: {batch_size}, Gestures: {len(gesture_recognitions)}, Landmarks: {len(hand_landmarks)}. Cannot proceed.")
            return (image,)
             
        # Pair gestures with landmarks for convenience
        paired_data = list(zip(gesture_recognitions, hand_landmarks))
        
        # Use the base class process_batch method with our custom drawing function
        return self.process_batch(
            paired_data,  # We're passing paired data instead of just detections
            image,
            self.draw_gestures,
            label_color=label_color,
            label_bg_color=label_bg_color,
            y_offset=y_offset
        )
    
    def draw_gestures(self, image, width, height, paired_data,
                      label_color="#FFFFFF", label_bg_color="#0000FF", y_offset=-30):
        """Draw gestures for a single image."""
        gestures_for_image, landmarks_for_image = paired_data
        
        # Check if number of detected hands matches for gestures and landmarks
        if len(gestures_for_image) != len(landmarks_for_image):
             logger.warning(f"VisualizeGestureRecognitions: Mismatch between detected gestures and landmarks. Skipping labels.")
             return
             
        if not gestures_for_image:
            return

        # Iterate through each detected hand/gesture set for this image
        for hand_idx, gesture_result in enumerate(gestures_for_image):
            landmark_result = landmarks_for_image[hand_idx]
            
            # Get top gesture category
            top_gesture = "N/A"
            if gesture_result.gestures:
                top_gesture = gesture_result.gestures[0].display_name or gesture_result.gestures[0].category_name or "N/A"
            
            # Use wrist landmark (index 0) for positioning text
            if not landmark_result.landmarks or len(landmark_result.landmarks) == 0:
                logger.warning(f"VisualizeGestureRecognitions: No landmarks found for this hand to position text.")
                continue
            
            wrist_lm = landmark_result.landmarks[0] # Assuming index 0 is wrist
            cx = int(wrist_lm.x * width)
            cy = int(wrist_lm.y * height) + y_offset # Apply offset

            label_text = f"{top_gesture}"
            
            # Draw label with background
            try:
                # Get text size
                text_width, text_height = self.get_text_dimensions(label_text, self.FONT_SCALE)
                
                # Center the text around the wrist point
                text_x = cx - (text_width // 2)
                text_y = cy - (text_height // 2)
                
                # Draw the text with background
                self.draw_text_with_background(
                    image,
                    label_text,
                    (text_x, text_y),
                    text_color=label_color,
                    bg_color=label_bg_color,
                    font_scale=self.FONT_SCALE
                )
            except Exception as e:
                logger.error(f"Error drawing text label for gesture: {e}")
                # Fallback: Draw text directly
                cv2.putText(
                    image, 
                    label_text, 
                    (cx, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    self.FONT_SCALE, 
                    self._hex_to_bgr(label_color), 
                    1, 
                    cv2.LINE_AA
                )

# Node registration
NODE_CLASS_MAPPINGS = {
    "VisualizeGestureRecognitions": VisualizeGestureRecognitions
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisualizeGestureRecognitions": "Visualize Gestures (MediaPipe)"
} 