# visualize_face.py
import logging
from typing import Any, List, Optional

import mediapipe as mp

from .....src.mediapipe_vision.common.base_visualization_nodes import BaseLandmarkVisualizationNode

logger = logging.getLogger(__name__)

# Define standard face connections
FACE_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_TESSELATION


class VisualizeFaceLandmarks(BaseLandmarkVisualizationNode):
    LANDMARKS_TYPE = "FACE_LANDMARKS"
    CONNECTIONS = FACE_CONNECTIONS
    DEFAULT_DRAW_POINTS = False
    DEFAULT_POINT_COLOR = "#FFFF00"
    DEFAULT_POINT_RADIUS = 1
    DEFAULT_CONNECTION_COLOR = "#00FF00"
    DEFAULT_CONNECTION_THICKNESS = 1
    SUPPORTS_VISIBILITY = False
    CATEGORY = "Realtime Nodes/MediaPipe Vision/Face/FaceLandmark/Visualization"
    DESCRIPTION = "Visualizes facial landmarks by drawing a mesh of points and connections on the face. Creates a detailed facial wireframe that shows the precise contours and features of detected faces."

    def get_landmark_list(self, detection_result: Any) -> Optional[List[Any]]:
        return detection_result


# Node registration
NODE_CLASS_MAPPINGS = {"VisualizeFaceLandmarks": VisualizeFaceLandmarks}

NODE_DISPLAY_NAME_MAPPINGS = {"VisualizeFaceLandmarks": "Visualize Face Landmarks (MediaPipe)"}
