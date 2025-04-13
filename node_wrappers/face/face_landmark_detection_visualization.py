# visualize_face.py
import mediapipe as mp
from typing import Optional, List, Any

from ..common.base_visualization_nodes import BaseLandmarkVisualizationNode

import logging
logger = logging.getLogger(__name__)

# Define standard face connections
FACE_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_TESSELATION

class VisualizeFaceLandmarks(BaseLandmarkVisualizationNode):
    """Draws Face landmarks and connections/contours onto images."""
    
    # Override base class properties
    LANDMARKS_TYPE = "FACE_LANDMARKS"
    CONNECTIONS = FACE_CONNECTIONS
    DEFAULT_DRAW_POINTS = False
    DEFAULT_POINT_COLOR = "#FFFF00"
    DEFAULT_POINT_RADIUS = 1
    DEFAULT_CONNECTION_COLOR = "#00FF00"
    DEFAULT_CONNECTION_THICKNESS = 1
    SUPPORTS_VISIBILITY = False
    CATEGORY = "MediaPipeVision/Face/FaceLandmark/Visualization"
    DESCRIPTION = "Visualizes facial landmarks by drawing a mesh of points and connections on the face. Creates a detailed facial wireframe that shows the precise contours and features of detected faces."
    
    def get_landmark_list(self, detection_result: Any) -> Optional[List[Any]]:
        """For face landmarks, the detection result itself is the list of landmarks."""
        # detection_result is expected to be List[LandmarkPoint]
        if isinstance(detection_result, list):
            return detection_result
        else:
            logger.warning(f"Unexpected format for face detection result: {type(detection_result)}")
            return None

# Node registration
NODE_CLASS_MAPPINGS = {
    "VisualizeFaceLandmarks": VisualizeFaceLandmarks
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisualizeFaceLandmarks": "Visualize Face Landmarks (MediaPipe)"
} 