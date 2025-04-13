# visualize_pose.py
import mediapipe as mp
from typing import Optional, List, Any

from ...src.types import PoseLandmarksResult # Import specific result type if needed for extraction
from ..common.base_visualization_nodes import BaseLandmarkVisualizationNode

import logging
logger = logging.getLogger(__name__)
_category = "MediaPipeVision/Pose/Visualization"
# Define standard pose connections
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

class VisualizePoseLandmarks(BaseLandmarkVisualizationNode):
    """Draws Pose landmarks and connections onto images."""
    
    # Override base class properties
    LANDMARKS_TYPE = "POSE_LANDMARKS"
    CONNECTIONS = POSE_CONNECTIONS
    DEFAULT_DRAW_POINTS = True
    DEFAULT_POINT_COLOR = "#00FFFF"  # Cyan
    DEFAULT_POINT_RADIUS = 3
    DEFAULT_CONNECTION_COLOR = "#FFFFFF"  # White
    DEFAULT_CONNECTION_THICKNESS = 2
    SUPPORTS_VISIBILITY = True # Pose landmarks have visibility
    CATEGORY = _category
    def get_landmark_list(self, detection_result: Any) -> Optional[List[Any]]:
        """Extracts the list of landmarks from a PoseLandmarksResult object."""
        if isinstance(detection_result, PoseLandmarksResult) and hasattr(detection_result, 'landmarks'):
            return detection_result.landmarks
        elif isinstance(detection_result, dict) and 'landmarks' in detection_result: # Handle potential dict format
             return detection_result['landmarks']
        else:
            logger.warning(f"Unexpected format for pose detection result: {type(detection_result)}")
            return None

# Node registration
NODE_CLASS_MAPPINGS = {
    "VisualizePoseLandmarks": VisualizePoseLandmarks
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisualizePoseLandmarks": "Visualize Pose Landmarks (MediaPipe)"
} 