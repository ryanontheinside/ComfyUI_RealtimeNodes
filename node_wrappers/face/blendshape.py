# node_wrappers/face/blendshape.py
import logging
from typing import Optional, List, Any, Dict, Tuple

from ...src.types import BLENDSHAPES_LIST, Blendshape # Updated type alias
from ...src.utils.misc_utils import get_blendshape_categories # Utility to get all category names

# Import base trigger/control logic if we adapt base classes later, or implement directly for now
from ..common.base_delta_nodes import FLOAT_EQUALITY_TOLERANCE # For trigger node comparisons
from ...src.utils.delta_utils import scale_value

logger = logging.getLogger(__name__)

# --- Constants ---
# Get the list of available blendshape categories from MediaPipe
ALL_BLENDSHAPE_CATEGORIES = get_blendshape_categories()
if not ALL_BLENDSHAPE_CATEGORIES:
    logger.warning("Could not retrieve blendshape categories. Blendshape nodes might not function correctly.")
    ALL_BLENDSHAPE_CATEGORIES = ["_neutral"] # Provide a fallback

_category = "MediaPipeVision/Face/Blendshape"


# --- Helper Functions ---
def get_blendshape_score(blendshapes_result: BLENDSHAPES_LIST, category_name: str) -> Optional[float]:
    """Extracts the score for a specific blendshape category from the result."""
    if not blendshapes_result or not blendshapes_result[0] or not blendshapes_result[0][0]:
        #logger.debug(f"No blendshapes found in result.")
        return None # No blendshape data

    # BLENDSHAPES structure: List[List[Dict[str, float]]]
    # We typically care about the first face in the first batch item
    first_face_blendshapes = blendshapes_result[0][0]

    # In the MediaPipe output, blendshapes are represented as a dictionary
    # where keys are category names and values are scores
    if isinstance(first_face_blendshapes, dict):
        # Dictionary format: {category_name: score, ...}
        if category_name in first_face_blendshapes:
            return first_face_blendshapes[category_name]
    elif isinstance(first_face_blendshapes, list):
        # Try to handle the list format with objects that have category_name attribute
        try:
            target_blendshape = next((bs for bs in first_face_blendshapes if hasattr(bs, 'category_name') and bs.category_name == category_name), None)
            if target_blendshape:
                return target_blendshape.score
        except Exception as e:
            logger.error(f"Error processing blendshape list: {e}")
    
    logger.warning(f"Blendshape category '{category_name}' not found in results. Type: {type(first_face_blendshapes)}")
    return None


# --- Control Nodes ---
class BlendshapeControlFloatNode:
    """Outputs a scaled FLOAT based on the score of a selected blendshape."""
    CATEGORY = _category
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    DESCRIPTION = "Converts facial expressions into floating point values that can control other nodes. For example, use a smile to control image parameters or animation intensity."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blendshapes": ("BLENDSHAPES_LIST", {"tooltip": "Facial expression data from the Face Landmarker node"}),
                "blendshape_name": (ALL_BLENDSHAPE_CATEGORIES, {"default": "mouthSmileLeft", 
                                                              "tooltip": "The specific facial expression to monitor (e.g., smile, frown, raised eyebrows)"}),
                # Blendshape scores are typically 0.0 to 1.0, but allow flexibility
                "score_min": ("FLOAT", {"default": 0.0, "step": 0.01, 
                                      "tooltip": "The minimum facial expression intensity to consider (usually 0.0)"}),
                "score_max": ("FLOAT", {"default": 1.0, "step": 0.01, 
                                      "tooltip": "The maximum facial expression intensity to consider (usually 1.0)"}),
                "output_min_float": ("FLOAT", {"default": 0.0, "step": 0.01, 
                                             "tooltip": "The minimum output value when expression is at or below score_min"}),
                "output_max_float": ("FLOAT", {"default": 1.0, "step": 0.01, 
                                             "tooltip": "The maximum output value when expression is at or above score_max"}),
                "clamp": ("BOOLEAN", {"default": True, 
                                    "tooltip": "When enabled, restricts output values to stay within min and max range"}),
            }
        }

    def execute(self, blendshapes, blendshape_name, score_min, score_max, output_min_float, output_max_float, clamp):
        raw_score = get_blendshape_score(blendshapes, blendshape_name)

        scaled_value = scale_value(raw_score, score_min, score_max, output_min_float, output_max_float, clamp)

        # Default to min value on error or if blendshape not found
        output_val = scaled_value if scaled_value is not None else float(output_min_float)
        return (output_val,)

class BlendshapeControlIntNode:
    """Outputs a scaled INT based on the score of a selected blendshape."""
    CATEGORY = _category
    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    DESCRIPTION = "Converts facial expressions into integer values that can control other nodes. Useful for selecting steps, seeds, or indexes based on facial expressions."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blendshapes": ("BLENDSHAPES_LIST", {"tooltip": "Facial expression data from the Face Landmarker node"}),
                "blendshape_name": (ALL_BLENDSHAPE_CATEGORIES, {"default": "mouthSmileLeft", 
                                                              "tooltip": "The specific facial expression to monitor (e.g., smile, frown, raised eyebrows)"}),
                "score_min": ("FLOAT", {"default": 0.0, "step": 0.01, 
                                      "tooltip": "The minimum facial expression intensity to consider (usually 0.0)"}),
                "score_max": ("FLOAT", {"default": 1.0, "step": 0.01, 
                                      "tooltip": "The maximum facial expression intensity to consider (usually 1.0)"}),
                "output_min_int": ("INT", {"default": 0, 
                                         "tooltip": "The minimum integer output value when expression is at or below score_min"}),
                "output_max_int": ("INT", {"default": 100, 
                                         "tooltip": "The maximum integer output value when expression is at or above score_max"}),
                "clamp": ("BOOLEAN", {"default": True, 
                                    "tooltip": "When enabled, restricts output values to stay within min and max range"}),
            }
        }

    def execute(self, blendshapes, blendshape_name, score_min, score_max, output_min_int, output_max_int, clamp):
        raw_score = get_blendshape_score(blendshapes, blendshape_name)

        scaled_value = scale_value(raw_score, score_min, score_max, output_min_int, output_max_int, clamp)

        # Default to min value on error or if blendshape not found
        output_val = int(round(scaled_value)) if scaled_value is not None else int(output_min_int)
        return (output_val,)


# --- Trigger Node ---
class BlendshapeTriggerNode:
    """Outputs a BOOLEAN trigger if a blendshape score crosses a threshold."""
    CATEGORY = _category
    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "execute"
    DESCRIPTION = "Creates a true/false trigger based on facial expressions exceeding thresholds. Perfect for activating workflows when someone smiles, frowns, or makes specific expressions."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blendshapes": ("BLENDSHAPES_LIST", {"tooltip": "Facial expression data from the Face Landmarker node"}),
                "blendshape_name": (ALL_BLENDSHAPE_CATEGORIES, {"default": "jawOpen", 
                                                              "tooltip": "The specific facial expression to monitor (e.g., smile, frown, open mouth)"}),
                "threshold": ("FLOAT", {"default": 0.5, "step": 0.01, 
                                      "tooltip": "The expression intensity threshold value that triggers the condition"}),
                "condition": (["Above", "Below", "Equals", "Not Equals"], 
                             {"tooltip": "The condition that determines when to trigger - Above (expression > threshold), Below (expression < threshold), etc."}),
            }
        }

    def execute(self, blendshapes, blendshape_name, threshold, condition):
        raw_score = get_blendshape_score(blendshapes, blendshape_name)

        triggered = False
        if raw_score is not None:
            if condition == "Above" and raw_score > threshold:
                triggered = True
            elif condition == "Below" and raw_score < threshold:
                triggered = True
            elif condition == "Equals" and abs(raw_score - threshold) < FLOAT_EQUALITY_TOLERANCE:
                triggered = True
            elif condition == "Not Equals" and abs(raw_score - threshold) >= FLOAT_EQUALITY_TOLERANCE:
                triggered = True

        return (triggered,)


# --- Mappings ---
NODE_CLASS_MAPPINGS = {
    "BlendshapeControlFloat": BlendshapeControlFloatNode,
    "BlendshapeControlInt": BlendshapeControlIntNode,
    "BlendshapeTrigger": BlendshapeTriggerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlendshapeControlFloat": "Blendshape Control (Float)",
    "BlendshapeControlInt": "Blendshape Control (Int)",
    "BlendshapeTrigger": "Blendshape Trigger",
} 