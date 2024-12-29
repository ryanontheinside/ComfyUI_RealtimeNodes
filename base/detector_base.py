from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

class Detector(ABC):
    """Base class for image region detectors"""
    
    @abstractmethod
    def setup(self, **kwargs):
        """Initialize detector with parameters"""
        pass
        
    @abstractmethod
    def detect(self, current_frame: np.ndarray, mask: np.ndarray, state: dict) -> tuple[float, np.ndarray]:
        """
        Analyze frame region and return detection value and visualization
        Args:
            current_frame: Current frame data
            mask: ROI mask
            state: Detector state dict for temporal data
        Returns:
            (detection_value, visualization_mask)
        """
        pass

class ROIAction(Enum):
    # Behavioral actions
    TOGGLE = "toggle"      # Toggles between min/max values
    MOMENTARY = "momentary"  # Outputs max while detected
    TRIGGER = "trigger"    # Triggers once per detection event
    COUNTER = "counter"    # Counts detection events
    # Mathematical actions
    ADD = "add"          # Add value when detected
    SUBTRACT = "subtract" # Subtract value when detected
    MULTIPLY = "multiply" # Multiply by value when detected
    DIVIDE = "divide"     # Divide by value when detected
    SET = "set"   