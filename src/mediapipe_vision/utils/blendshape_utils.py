# src/utils/misc_utils.py
import mediapipe as mp
import logging
from typing import List

logger = logging.getLogger(__name__)

def get_blendshape_categories() -> List[str]:
    """Returns the list of standard blendshape category names from MediaPipe."""
    # Define blendshape categories directly 
    # This is based on the MediaPipe standard blendshapes
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
    
    logger.info("Using predefined list of standard blendshape categories.")
    return standard_blendshapes 