# src/utils/misc_utils.py
import mediapipe as mp
import logging
from typing import List

logger = logging.getLogger(__name__)

def get_blendshape_categories() -> List[str]:
    """Returns the list of standard blendshape category names from MediaPipe."""
    # MediaPipe standard blendshape categories
    # Hardcoded list based on MediaPipe's standard blendshapes
    # This avoids dependency on specific MediaPipe constant locations that may change
    standard_blendshapes = [
        "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", 
        "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", 
        "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", 
        "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", 
        "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", 
        "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", 
        "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", 
        "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", 
        "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", 
        "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower", 
        "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", 
        "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", 
        "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"
    ]
    
    try:
        # First attempt to dynamically get from MediaPipe if possible (future-proofing)
        # This section tries to access newest MediaPipe API first
        blendshape_tuples = mp.solutions.face_mesh.FACEMESH_BLENDSHAPES
        category_names = sorted([name for index, name in blendshape_tuples])
        if category_names:
            logger.info(f"Successfully retrieved {len(category_names)} blendshape categories from MediaPipe.")
            return category_names
        else:
            logger.warning("MediaPipe FACEMESH_BLENDSHAPES constant returned empty list.")
            logger.info("Using hardcoded list of standard blendshape categories instead.")
            return standard_blendshapes
    except AttributeError:
        # Log the error but use our hardcoded list
        logger.warning("Could not find FACEMESH_BLENDSHAPES constant at expected path in mediapipe.solutions.face_mesh.")
        logger.info("Using hardcoded list of standard blendshape categories instead.")
        return standard_blendshapes
    except Exception as e:
        logger.error(f"An unexpected error occurred while getting blendshape categories: {e}")
        logger.info("Using hardcoded list of standard blendshape categories instead.")
        return standard_blendshapes 