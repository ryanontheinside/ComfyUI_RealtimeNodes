import logging
import mediapipe as mp
import numpy as np
import cv2
from typing import List, Dict, Tuple, Any, Optional
import torch

logger = logging.getLogger(__name__)
_category = "MediaPipeVision/Holistic"

# Use MediaPipe's standard connection definitions instead of defining our own
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
FACE_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_TESSELATION

class MediaPipeHolisticVisualizerNode:
    """ComfyUI node for visualizing holistic landmarks using direct drawing.
    
    Since holistic landmarks combine multiple landmark types (face, pose, hands),
    this doesn't use the BaseLandmarkVisualizationNode but implements a similar approach.
    """
    
    CATEGORY = _category
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "visualize_holistic"
    DESCRIPTION = "Visualizes holistic landmarks by drawing face mesh, pose skeleton, and hand landmarks on the input image."
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image on which to visualize landmarks"}),
                "landmarks": ("HOLISTIC_LANDMARKS", {"tooltip": "Landmarks from the holistic detector"}),
                "face_color": ("STRING", {"default": "#FF3030", "tooltip": "Color for face landmarks"}),
                "pose_color": ("STRING", {"default": "#30FF30", "tooltip": "Color for pose landmarks"}),
                "left_hand_color": ("STRING", {"default": "#3030FF", "tooltip": "Color for left hand landmarks"}),
                "right_hand_color": ("STRING", {"default": "#FF30FF", "tooltip": "Color for right hand landmarks"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Toggle drawing face landmarks"}),
                "draw_pose": ("BOOLEAN", {"default": True, "tooltip": "Toggle drawing pose landmarks"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Toggle drawing hand landmarks"}),
                "face_point_radius": ("INT", {"default": 1, "min": 1, "max": 5, "tooltip": "Size of face landmark points"}),
                "pose_point_radius": ("INT", {"default": 2, "min": 1, "max": 10, "tooltip": "Size of pose landmark points"}),
                "hand_point_radius": ("INT", {"default": 2, "min": 1, "max": 10, "tooltip": "Size of hand landmark points"}),
                "connection_thickness": ("INT", {"default": 1, "min": 1, "max": 5, "tooltip": "Thickness of connection lines"}),
                "min_visibility": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, 
                                           "tooltip": "Minimum visibility score to draw a pose landmark"})
            }
        }
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color string to RGB tuple for OpenCV (BGR format)."""
        hex_color = hex_color.lstrip('#')
        # Convert to BGR for OpenCV
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)  # Return in BGR format for OpenCV
    
    def visualize_holistic(self, 
                         image,
                         landmarks, 
                         face_color, 
                         pose_color, 
                         left_hand_color, 
                         right_hand_color,
                         draw_face, 
                         draw_pose, 
                         draw_hands,
                         face_point_radius,
                         pose_point_radius,
                         hand_point_radius,
                         connection_thickness,
                         min_visibility):
        """Visualize holistic landmarks using OpenCV for faster performance."""
        
        if not landmarks:
            logger.warning("No holistic landmarks provided.")
            return (image,)

        batch_size = image.shape[0]
        result_images = []

        # For holistic, each item in the batch is already a single detection
        # (unlike other landmarks which have a list of detections per image)
        if len(landmarks) != batch_size:
            logger.warning(f"Batch size mismatch. Images: {batch_size}, Landmark sets: {len(landmarks)}. Proceeding with available data.")

        # Convert color strings to BGR tuples for OpenCV
        face_color_bgr = self._hex_to_rgb(face_color)
        pose_color_bgr = self._hex_to_rgb(pose_color)
        left_hand_color_bgr = self._hex_to_rgb(left_hand_color)
        right_hand_color_bgr = self._hex_to_rgb(right_hand_color)
        
        for i in range(batch_size):
            # Clone the tensor to avoid modifying the original
            img_tensor = image[i].clone() 
            
            # Convert tensor to numpy array scaled to 0-255 range for OpenCV
            np_image = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            
            # Convert to BGR for OpenCV processing
            cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            
            if i < len(landmarks):
                holistic_result = landmarks[i]  # One holistic result per image
                
                # Draw face landmarks if requested
                if draw_face and holistic_result.face_landmarks:
                    self._draw_landmarks_cv2(
                        cv_image,
                        holistic_result.face_landmarks, 
                        FACE_CONNECTIONS,
                        face_color_bgr, face_point_radius, connection_thickness,
                        min_visibility=None  # Face doesn't use visibility
                    )
                
                # Draw pose landmarks if requested
                if draw_pose and holistic_result.pose_landmarks:
                    self._draw_landmarks_cv2(
                        cv_image,
                        holistic_result.pose_landmarks, 
                        POSE_CONNECTIONS,
                        pose_color_bgr, pose_point_radius, connection_thickness,
                        min_visibility=min_visibility
                    )
                
                # Draw left hand landmarks if requested
                if draw_hands and holistic_result.left_hand_landmarks:
                    self._draw_landmarks_cv2(
                        cv_image,
                        holistic_result.left_hand_landmarks, 
                        HAND_CONNECTIONS,
                        left_hand_color_bgr, hand_point_radius, connection_thickness,
                        min_visibility=None  # Hands don't use visibility
                    )
                
                # Draw right hand landmarks if requested
                if draw_hands and holistic_result.right_hand_landmarks:
                    self._draw_landmarks_cv2(
                        cv_image,
                        holistic_result.right_hand_landmarks, 
                        HAND_CONNECTIONS,
                        right_hand_color_bgr, hand_point_radius, connection_thickness,
                        min_visibility=None  # Hands don't use visibility
                    )
            
            # Convert back to RGB and tensor format
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            result_tensor = torch.from_numpy(cv_image_rgb.astype(np.float32) / 255.0)
            result_images.append(result_tensor)
        
        return (torch.stack(result_images),)
    
    def _draw_landmarks_cv2(self, 
                          image: np.ndarray, 
                          landmark_list: List,
                          connections: List,
                          color: Tuple[int, int, int],
                          point_radius: int,
                          connection_thickness: int,
                          min_visibility: Optional[float] = None):
        """Helper method to draw landmarks with OpenCV for faster performance."""
        
        height, width = image.shape[:2]
        points = {}
        
        # Process landmarks and draw points
        for lm in landmark_list:
            # Check visibility if required
            is_visible = True
            if min_visibility is not None and hasattr(lm, 'visibility') and lm.visibility is not None:
                is_visible = lm.visibility >= min_visibility
                
            if is_visible:
                # Convert normalized coordinates to pixel coordinates
                cx, cy = int(lm.x * width), int(lm.y * height)
                points[lm.index] = (cx, cy)
                
                # Draw point as a filled circle
                cv2.circle(image, (cx, cy), point_radius, color, -1)
        
        # Draw connections
        if connections and points:
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx in points and end_idx in points:
                    cv2.line(
                        image, 
                        points[start_idx], 
                        points[end_idx], 
                        color, 
                        connection_thickness
                    )


# Define mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MediaPipeHolisticVisualizer": MediaPipeHolisticVisualizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeHolisticVisualizer": "Holistic Visualizer (MediaPipe)",
} 