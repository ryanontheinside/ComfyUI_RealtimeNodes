from abc import ABC, abstractmethod
import numpy as np
import cv2
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from base.detector_base import Detector

class MotionDetector(Detector):
    def setup(self, threshold=0.1, blur_size=5):
        self.threshold = threshold
        self.blur_size = blur_size
        self.prev_frame = None
    
    @classmethod
    def preprocess(cls, frame: np.ndarray, shared_data: dict) -> dict:
        """Use shared grayscale and blurred data"""
        return {
            "blurred": shared_data["blurred"]  # Reuse shared blurred frame
        }
    
    def detect(self, current_frame: np.ndarray, mask: np.ndarray, state: dict) -> tuple[float, np.ndarray]:
        # Get preprocessed data
        preprocessed = state.get("preprocessed", {})
        shared = state.get("shared", {})
        y_offset = state.get('y_offset', 0)
        x_offset = state.get('x_offset', 0)
        
        # Get the blurred ROI from preprocessed data
        full_blurred = preprocessed["blurred"]
        # Extract the correct ROI region from the full blurred image
        current_blurred = full_blurred[y_offset:y_offset+current_frame.shape[0], 
                                     x_offset:x_offset+current_frame.shape[1]]
        
        # Initialize or update previous frame for this ROI
        roi_key = f"{y_offset},{x_offset}"
        if "prev_blurred" not in state:
            state["prev_blurred"] = {}
        if roi_key not in state["prev_blurred"]:
            state["prev_blurred"][roi_key] = current_blurred
            return 0.0, np.zeros_like(current_blurred, dtype=np.float32)
            
        # Calculate frame difference
        diff = cv2.absdiff(current_blurred, state["prev_blurred"][roi_key])
        _, thresh = cv2.threshold(diff, self.threshold * 255, 255, cv2.THRESH_BINARY)
        
        # Apply ROI mask
        thresh = thresh * (mask > 0.5)
        
        # Update previous frame
        state["prev_blurred"][roi_key] = current_blurred.copy()
        
        # Calculate motion magnitude
        roi_area = np.sum(mask > 0.5)
        if roi_area > 0:
            motion_magnitude = np.sum(thresh) / (255.0 * roi_area)
        else:
            motion_magnitude = 0.0
            
        return motion_magnitude, thresh.astype(np.float32) / 255.0

class BrightnessDetector(Detector):
    def setup(self, threshold: float = 0.5, use_average: bool = True):
        self.threshold = threshold
        self.use_average = use_average
    
    @classmethod
    def preprocess(cls, frame: np.ndarray, shared_data: dict) -> dict:
        """Use shared grayscale data"""
        return {
            "gray": shared_data["gray"]  # Reuse shared grayscale frame
        }
    
    def detect(self, current_frame: np.ndarray, mask: np.ndarray, state: dict) -> tuple[float, np.ndarray]:
        # Get preprocessed data
        preprocessed = state.get("preprocessed", {})
        shared = state.get("shared", {})
        y_offset = state.get('y_offset', 0)
        x_offset = state.get('x_offset', 0)
        
        # Get the grayscale ROI from preprocessed data
        full_gray = preprocessed["gray"]
        # Extract the correct ROI region from the full grayscale image
        gray = full_gray[y_offset:y_offset+current_frame.shape[0], 
                        x_offset:x_offset+current_frame.shape[1]]
        
        # Apply mask
        masked_gray = gray * (mask > 0.5)
        
        # Calculate brightness
        if self.use_average:
            brightness = np.mean(masked_gray) / 255.0
        else:
            brightness = np.max(masked_gray) / 255.0
            
        # Create visualization
        viz_mask = (masked_gray > (self.threshold * 255)).astype(np.float32)
        
        return brightness, viz_mask

# Registry of available detectors
DETECTORS = {
    "motion": MotionDetector,
    "brightness": BrightnessDetector
} 