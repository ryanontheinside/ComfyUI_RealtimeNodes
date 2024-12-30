from abc import ABC, abstractmethod
import numpy as np
import cv2
from ..base.detector_base import Detector

class MotionDetector(Detector):
    @classmethod
    def preprocess(cls, current_frame):
        """Preprocess the entire frame for motion detection"""
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        print(f"Preprocessed frame shape: {curr_gray.shape}")
        return {
            'gray_frame': curr_gray
        }
    
    def setup(self, threshold: float = 0.1, blur_size: int = 5):
        self.threshold = threshold
        self.blur_size = blur_size
    
    def detect(self, current_frame: np.ndarray, mask: np.ndarray, state: dict) -> tuple[float, np.ndarray]:
        # Log input shapes and ROI info
        print(f"\nMotion Detector:")
        print(f"ROI frame shape: {current_frame.shape}")
        print(f"ROI mask shape: {mask.shape}")
        print(f"State keys: {list(state.keys())}")
        
        # Use preprocessed grayscale frame if available
        preprocessed = state.get('preprocessed', {})
        if 'gray_frame' in preprocessed:
            full_gray = preprocessed['gray_frame']
            print(f"Full preprocessed gray shape: {full_gray.shape}")
            
            # Get the ROI slice from the preprocessed frame using the correct offset
            # The ROI coordinates are stored in the state or can be derived from the frame shape
            y_offset = state.get('y_offset', 0)  # We need to pass these from the control base
            x_offset = state.get('x_offset', 0)
            print(f"Using offsets: y={y_offset}, x={x_offset}")
            
            curr_gray = full_gray[y_offset:y_offset + current_frame.shape[0], 
                                x_offset:x_offset + current_frame.shape[1]]
            print(f"Sliced gray ROI shape: {curr_gray.shape}")
        else:
            print("No preprocessed data found, computing grayscale for ROI")
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        
        # Apply blur
        curr_blur = cv2.GaussianBlur(curr_gray, (self.blur_size, self.blur_size), 0)
        
        # Get previous frame from state
        prev_blur = state.get('prev_frame_blurred')
        if prev_blur is None:
            print("First frame, initializing previous frame")
            state['prev_frame_blurred'] = curr_blur
            return 0.0, np.zeros_like(curr_gray, dtype=np.float32)
        
        # Calculate motion
        diff = cv2.absdiff(curr_blur, prev_blur)
        _, thresh = cv2.threshold(diff, self.threshold * 255, 255, cv2.THRESH_BINARY)
        thresh = thresh * (mask > 0.5)
        
        # Calculate detection value
        roi_area = np.sum(mask > 0.5)
        if roi_area > 0:
            detection = np.sum(thresh) / (255.0 * roi_area)
        else:
            detection = 0.0
            
        # Update state
        state['prev_frame_blurred'] = curr_blur
        
        print(f"Motion detection value: {detection:.3f}")
        print(f"ROI area: {roi_area}")
        print(f"Sum of motion pixels: {np.sum(thresh)}")
        
        return detection, thresh / 255.0

class BrightnessDetector(Detector):
    @classmethod
    def preprocess(cls, current_frame):
        """Preprocess the entire frame for brightness detection"""
        # Convert to grayscale once
        gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        return {
            'gray_frame': gray
        }
    
    def setup(self, threshold: float = 0.5, use_average: bool = True):
        self.threshold = threshold
        self.use_average = use_average
    
    def detect(self, current_frame: np.ndarray, mask: np.ndarray, state: dict) -> tuple[float, np.ndarray]:
        # Debug: Log state contents
        print(f"Detector state keys: {list(state.keys())}")
        if 'preprocessed' in state:
            print(f"Preprocessed data keys: {list(state['preprocessed'].keys())}")
        
        # Use preprocessed grayscale frame if available
        preprocessed = state.get('preprocessed', {})
        if 'gray_frame' in preprocessed:
            # Use the pre-computed grayscale frame
            gray = preprocessed['gray_frame'][
                current_frame.shape[0]:current_frame.shape[0] + current_frame.shape[0],
                current_frame.shape[1]:current_frame.shape[1] + current_frame.shape[1]
            ]
        else:
            # Fallback to computing grayscale for this region
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