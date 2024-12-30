from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
import cv2

class SharedProcessing:
    """Handles common image processing operations used by multiple detectors"""
    
    @staticmethod
    def get_shared_data(frame: np.ndarray, blur_size: int = 5) -> dict:
        """
        Performs common image processing operations once
        Args:
            frame: RGB numpy array
            blur_size: Size of Gaussian blur kernel
        Returns:
            dict of preprocessed data
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        return {
            "gray": gray,
            "blurred": blurred,
            "frame": frame  # Original frame
        }

class Detector(ABC):
    """Base class for image region detectors"""
    
    @classmethod
    def preprocess(cls, frame: np.ndarray, shared_data: dict = None) -> dict:
        """
        Optional preprocessing step for the entire frame
        Args:
            frame: Full frame data as numpy array
            shared_data: Dict containing shared preprocessed data
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
                - preprocessed: dict of preprocessed data from preprocess()
                - shared: dict of shared preprocessed data
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
    SET = "set"          # Set to value when detected

# Define descriptions as a separate dictionary
ROIAction.descriptions = {
    ROIAction.TOGGLE.value: "Toggles between min/max values on each detection",
    ROIAction.MOMENTARY.value: "Outputs max value while detection is active, min otherwise",
    ROIAction.TRIGGER.value: "Outputs max value once per detection event",
    ROIAction.COUNTER.value: "Counts the number of detection events",
    ROIAction.ADD.value: "Adds the specified value when detection occurs",
    ROIAction.SUBTRACT.value: "Subtracts the specified value when detection occurs",
    ROIAction.MULTIPLY.value: "Multiplies by the specified value when detection occurs",
    ROIAction.DIVIDE.value: "Divides by the specified value when detection occurs",
    ROIAction.SET.value: "Sets output directly to the specified value when detection occurs"
} 