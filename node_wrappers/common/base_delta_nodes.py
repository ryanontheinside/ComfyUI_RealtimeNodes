# node_wrappers/common/base_delta_nodes.py
import logging
import collections # Added collections
from typing import Optional, Callable, Any, Tuple, Dict, Type

from ...src.utils.delta_utils import calculate_euclidean_delta, scale_value # Added scale_value import
# from ..common.delta_logic import execute_delta_control, execute_delta_trigger

logger = logging.getLogger(__name__)

# --- Constants ---
FLOAT_EQUALITY_TOLERANCE = 1e-6
MAX_LANDMARK_HISTORY = 50 # Max frames history for window/velocity

# --- Abstract Delta Calculation Helper ---
# This pattern avoids repeating the same extraction logic in each feature's delta file
def get_landmark_delta(landmarks_current: Any, landmarks_previous: Any, landmark_index: int,
                       get_landmarks_list_func: Callable[[Any], Optional[list]]) -> Optional[float]:
    """Generic helper to extract specific landmark points and calculate delta."""
    
    list_current = get_landmarks_list_func(landmarks_current)
    list_previous = get_landmarks_list_func(landmarks_previous)

    if not list_current or not list_previous:
        return None # Could happen if no faces/hands/poses were detected in current/previous

    point_current = next((lm for lm in list_current if lm.index == landmark_index), None)
    point_previous = next((lm for lm in list_previous if lm.index == landmark_index), None)
    
    if point_current is None or point_previous is None: 
         # Reduced logging severity as this can happen normally if landmark isn't always present
         # logger.warning(f"Landmark index {landmark_index} not found in current or previous landmarks.")
         return None

    return calculate_euclidean_delta(point_current, point_previous)


# --- Base Node Classes ---
class BaseDeltaNode:
    """Common base class for delta nodes to handle landmark history storage."""
    FUNCTION = "execute"
    
    # Must be overridden
    CATEGORY = "MediaPipeVision/Base"
    LANDMARKS_TYPE = "ANY"
    DEFAULT_LANDMARK_INDEX = 0
    MIN_LANDMARK_INDEX = 0
    MAX_LANDMARK_INDEX = 0
    TOOLTIP = ""

    def __init__(self):
        # Use a deque to store landmark history for windowing
        self._landmark_history = collections.deque(maxlen=MAX_LANDMARK_HISTORY)
        
    def get_raw_delta(self, current_landmarks: Any, landmark_index: int, window_size: int = 1) -> Optional[float]:
        """Calculates the raw delta compared to landmarks from 'window_size' frames ago."""
        
        # Ensure window size is valid for deque and logic
        window_size = max(1, min(window_size, MAX_LANDMARK_HISTORY - 1))
        
        # Store current landmarks
        self._landmark_history.append(current_landmarks)
        
        # Check if we have enough history for the requested window
        # Need window_size + 1 elements to compare index -1 and index -(window_size + 1)
        if len(self._landmark_history) > window_size:
            landmarks_now = self._landmark_history[-1]
            landmarks_past = self._landmark_history[-1 - window_size] # Get element from window_size steps ago
            
            if landmarks_now is None or landmarks_past is None:
                # This might happen if detection failed on some frames in the history
                logger.debug(f"{self.__class__.__name__}: Missing landmark data in history window.")
                return None
                
            raw_delta = get_landmark_delta(
                landmarks_now,
                landmarks_past, 
                landmark_index,
                self.get_landmarks_list # Use the specific feature's landmark list extractor
            )
            return raw_delta
        else:
            logger.debug(f"{self.__class__.__name__}: Not enough landmark history for window size {window_size}. Need {window_size + 1}, have {len(self._landmark_history)}.")
            return None # Not enough history yet

    def get_landmarks_list(self, landmarks_result: Any) -> Optional[list]:
        """Must be implemented by subclasses to extract the list of Landmark objects
           from the specific landmark result type (e.g., FaceLandmarksResult[0][0])."""
        raise NotImplementedError("Subclasses must implement get_landmarks_list")


class BaseLandmarkDeltaControlNode(BaseDeltaNode):
    """Base class for landmark delta control nodes (Int/Float)."""
    # Must be overridden by subclasses (Int or Float)
    RETURN_TYPES = ("FLOAT",)
    IS_INT_OUTPUT = False
    DEFAULT_DELTA_MIN = 0.0
    DEFAULT_DELTA_MAX = 0.1
    DEFAULT_OUTPUT_MIN = 0.0
    DEFAULT_OUTPUT_MAX = 1.0
    DEFAULT_CLAMP = True

    @classmethod
    def INPUT_TYPES(cls):
        output_type_str = "INT" if cls.IS_INT_OUTPUT else "FLOAT"
        output_min_key = f"output_min_{output_type_str.lower()}"
        output_max_key = f"output_max_{output_type_str.lower()}"
        
        return {
            "required": {
                "landmarks": (cls.LANDMARKS_TYPE,),
                "landmark_index": ("INT", {
                    "default": cls.DEFAULT_LANDMARK_INDEX,
                    "min": cls.MIN_LANDMARK_INDEX,
                    "max": cls.MAX_LANDMARK_INDEX,
                    "tooltip": cls.TOOLTIP
                }),
                "window_size": ("INT", { # Added window_size
                    "default": 1, 
                    "min": 1, 
                    "max": MAX_LANDMARK_HISTORY - 1, # Limit to deque capacity
                    "tooltip": "Number of frames back to compare for delta (1 = vs previous frame)"
                }), 
                "delta_min": ("FLOAT", {"default": cls.DEFAULT_DELTA_MIN, "step": 0.01}),
                "delta_max": ("FLOAT", {"default": cls.DEFAULT_DELTA_MAX, "step": 0.01}),
                output_min_key: (output_type_str, {"default": cls.DEFAULT_OUTPUT_MIN}),
                output_max_key: (output_type_str, {"default": cls.DEFAULT_OUTPUT_MAX}),
                "clamp": ("BOOLEAN", {"default": cls.DEFAULT_CLAMP}),
            }
        }

    def execute(self, **kwargs):
        # Determine dynamic keys
        output_type_str = "INT" if self.IS_INT_OUTPUT else "FLOAT"
        output_min_key = f"output_min_{output_type_str.lower()}"
        output_max_key = f"output_max_{output_type_str.lower()}"
        
        # Extract arguments from kwargs
        landmarks = kwargs.get('landmarks')
        landmark_index = kwargs.get('landmark_index')
        window_size = kwargs.get('window_size', 1) # Extract window_size
        delta_min = kwargs.get('delta_min')
        delta_max = kwargs.get('delta_max')
        output_min = kwargs.get(output_min_key)
        output_max = kwargs.get(output_max_key)
        clamp = kwargs.get('clamp')
        
        # Basic validation in case kwargs are missing expected keys (shouldn't happen with ComfyUI)
        if None in [landmarks, landmark_index, delta_min, delta_max, output_min, output_max, clamp]:
            # Log an error or raise an exception if needed
            logger.error(f"{self.__class__.__name__}: Missing required arguments in execute method.")
            # Decide how to handle this - return default or raise error?
            # For now, let's try to return a default based on output type
            fallback_value = int(self.DEFAULT_OUTPUT_MIN) if self.IS_INT_OUTPUT else float(self.DEFAULT_OUTPUT_MIN)
            return (fallback_value,)

        # Get delta using the window size
        raw_delta = self.get_raw_delta(landmarks, landmark_index, window_size)
        
        # Note: No swapping logic added for min > max, relying on scale_value
        scaled_value = scale_value(raw_delta, delta_min, delta_max, output_min, output_max, clamp)
        
        if self.IS_INT_OUTPUT:
            # Default to min value on error or first frame
            output_val = int(round(scaled_value)) if scaled_value is not None else int(output_min)
            return (output_val,)
        else:
            # Default to min value on error or first frame
            output_val = scaled_value if scaled_value is not None else float(output_min)
            return (output_val,)


class BaseLandmarkDeltaIntControlNode(BaseLandmarkDeltaControlNode):
    RETURN_TYPES = ("INT",)
    IS_INT_OUTPUT = True
    DEFAULT_OUTPUT_MIN = 0
    DEFAULT_OUTPUT_MAX = 100


class BaseLandmarkDeltaFloatControlNode(BaseLandmarkDeltaControlNode):
    RETURN_TYPES = ("FLOAT",)
    IS_INT_OUTPUT = False
    DEFAULT_OUTPUT_MIN = 0.0
    DEFAULT_OUTPUT_MAX = 1.0


class BaseLandmarkDeltaTriggerNode(BaseDeltaNode):
    """Base class for landmark delta trigger nodes."""
    RETURN_TYPES = ("BOOLEAN",)
    DEFAULT_THRESHOLD = 0.05

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "landmarks": (cls.LANDMARKS_TYPE,),
                "landmark_index": ("INT", {
                    "default": cls.DEFAULT_LANDMARK_INDEX,
                    "min": cls.MIN_LANDMARK_INDEX,
                    "max": cls.MAX_LANDMARK_INDEX,
                    "tooltip": cls.TOOLTIP
                }),
                "window_size": ("INT", { # Added window_size
                    "default": 1, 
                    "min": 1, 
                    "max": MAX_LANDMARK_HISTORY - 1, # Limit to deque capacity
                    "tooltip": "Number of frames back to compare for delta (1 = vs previous frame)"
                }), 
                "threshold": ("FLOAT", {"default": cls.DEFAULT_THRESHOLD, "step": 0.01}),
                # Added Equals and Not Equals conditions
                "condition": (["Above", "Below", "Equals", "Not Equals"],),
            }
        }

    def execute(self, **kwargs):
        # Extract arguments
        landmarks = kwargs.get('landmarks')
        landmark_index = kwargs.get('landmark_index')
        window_size = kwargs.get('window_size', 1) # Extract window_size
        threshold = kwargs.get('threshold')
        condition = kwargs.get('condition')

        # Basic validation
        if None in [landmarks, landmark_index, threshold, condition]:
             logger.error(f"{self.__class__.__name__}: Missing required arguments in execute method.")
             return (False,) # Default to False if inputs missing

        # Get delta using the window size
        raw_delta = self.get_raw_delta(landmarks, landmark_index, window_size)

        triggered = False
        if raw_delta is not None:
            if condition == "Above" and raw_delta > threshold:
                triggered = True
            elif condition == "Below" and raw_delta < threshold:
                triggered = True
            # Handle Equals with tolerance
            elif condition == "Equals" and abs(raw_delta - threshold) < FLOAT_EQUALITY_TOLERANCE:
                triggered = True
            # Handle Not Equals with tolerance
            elif condition == "Not Equals" and abs(raw_delta - threshold) >= FLOAT_EQUALITY_TOLERANCE:
                 triggered = True
                
        return (triggered,) 