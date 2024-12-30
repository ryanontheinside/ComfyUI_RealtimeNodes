from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

class Detector(ABC):
    """Base class for image region detectors"""
    
    @classmethod
    def preprocess(cls, frame: np.ndarray) -> dict:
        """
        Optional preprocessing step for the entire frame
        Args:
            frame: Full frame data as numpy array
        Returns:
            dict containing preprocessed data to be shared across ROIs
        """
        return {}
        
    @abstractmethod
    def setup(self, **kwargs):
        """Initialize detector with parameters"""
        pass
        
    @abstractmethod
    def detect(self, current_frame: np.ndarray, mask: np.ndarray, state: dict) -> tuple[float, np.ndarray]:
        """
        Analyze frame region and return detection value and visualization
        Args:
            current_frame: Current frame data for ROI region
            mask: ROI mask
            state: Detector state dict containing:
                - preprocessed: dict of preprocessed data from preprocess() if available
                - Any other temporal or persistent data needed by the detector
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