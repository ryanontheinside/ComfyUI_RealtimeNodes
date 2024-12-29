from abc import ABC, abstractmethod
import numpy as np
import cv2
from ..base.detector_base import Detector

class MotionDetector(Detector):
    def setup(self, threshold: float = 0.1, blur_size: int = 5):
        self.threshold = threshold
        self.blur_size = blur_size
    
    def detect(self, current_frame: np.ndarray, mask: np.ndarray, state: dict) -> tuple[float, np.ndarray]:
        # Convert and blur current frame
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        curr_blur = cv2.GaussianBlur(curr_gray, (self.blur_size, self.blur_size), 0)
        
        # Get previous frame from state
        prev_blur = state.get('prev_frame_blurred')
        if prev_blur is None:
            state['prev_frame_blurred'] = curr_blur
            return 0.0, np.zeros_like(curr_gray, dtype=np.float32)
            
        # Calculate motion
        diff = cv2.absdiff(curr_blur, prev_blur)
        _, thresh = cv2.threshold(diff, self.threshold * 255, 255, cv2.THRESH_BINARY)
        thresh = thresh * (mask > 0.5)
        
        # Update state
        state['prev_frame_blurred'] = curr_blur
        
        # Calculate detection value
        roi_area = np.sum(mask > 0.5)
        if roi_area > 0:
            detection = np.sum(thresh) / (255.0 * roi_area)
        else:
            detection = 0.0
            
        return detection, thresh / 255.0

class BrightnessDetector(Detector):
    def setup(self, threshold: float = 0.5, use_average: bool = True):
        self.threshold = threshold
        self.use_average = use_average
    
    def detect(self, current_frame: np.ndarray, mask: np.ndarray, state: dict) -> tuple[float, np.ndarray]:
        # Convert to grayscale
        gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
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