# visualize_hand.py
import mediapipe as mp
from typing import Optional, List, Any

from ...src.types import HandLandmarksResult # Import specific result type if needed for extraction
from ..common.base_visualization_nodes import BaseLandmarkVisualizationNode

import logging
logger = logging.getLogger(__name__)
_category = "MediaPipeVision/Hand/HandLandmark/Visualization"
# Define standard hand connections
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

class VisualizeHandLandmarks(BaseLandmarkVisualizationNode):
    """Draws Hand landmarks and connections onto images."""
    
    # Override base class properties
    LANDMARKS_TYPE = "HAND_LANDMARKS"
    CONNECTIONS = HAND_CONNECTIONS
    DEFAULT_DRAW_POINTS = True
    DEFAULT_POINT_COLOR = "#FF0000"  # Red
    DEFAULT_POINT_RADIUS = 3
    DEFAULT_CONNECTION_COLOR = "#00FF00"  # Green
    DEFAULT_CONNECTION_THICKNESS = 2
    SUPPORTS_VISIBILITY = False
    CATEGORY = _category
    DESCRIPTION = "Visualizes hand landmarks by drawing points and connections on detected hands. Creates a wireframe overlay showing finger joints, palm, and hand structure."
    
    def get_landmark_list(self, detection_result: Any) -> Optional[List[Any]]:
        """Extracts the list of landmarks from a HandLandmarksResult object."""
        if isinstance(detection_result, HandLandmarksResult) and hasattr(detection_result, 'landmarks'):
            return detection_result.landmarks
        elif isinstance(detection_result, dict) and 'landmarks' in detection_result: # Handle potential dict format
            return detection_result['landmarks']
        else:
            logger.warning(f"Unexpected format for hand detection result: {type(detection_result)}")
            return None

# Node registration
NODE_CLASS_MAPPINGS = {
    "VisualizeHandLandmarks": VisualizeHandLandmarks
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisualizeHandLandmarks": "Visualize Hand Landmarks (MediaPipe)"
} 