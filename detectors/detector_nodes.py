import sys
import os
from enum import Enum
import numpy as np

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from detectors.detectors import MotionDetector, BrightnessDetector
from base.control_base import ControlNodeBase
from base.detector_base import ROIAction

class RegionOfInterest(ControlNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()  # Get base inputs
        
        # Format tooltip using ROIAction descriptions
        action_tooltip = "Available actions:\n" + "\n".join(
            f"• {action.value}: {ROIAction.descriptions[action.value]}" 
            for action in ROIAction
        )
        
        inputs["required"].update({
            "mask": ("MASK", {
                "tooltip": "Binary mask defining the region of interest"
            }),
            "detector": ("DETECTOR", {
                "tooltip": "Configured detector node (Motion or Brightness) to process this region"
            }),
            "action": (list(action.value for action in ROIAction), {
                "tooltip": action_tooltip
            }),
            "value": ("FLOAT", {
                "default": 0.1,
                "step": 0.01,
                "tooltip": "Value to use in the selected action (amount to add/subtract/multiply/divide/set)"
            }),
        })
        inputs["optional"] = {
            "next_roi": ("ROI", {
                "tooltip": "Optional connection to chain multiple ROIs together"
            })
        }
        return inputs

    RETURN_TYPES = ("ROI",)
    FUNCTION = "update"
    CATEGORY = "real-time/control/detection"

    def update(self, mask, detector, action, value, always_execute=False, next_roi=None):
        """Implements required update method from ControlNodeBase"""
        mask_np = mask[0].cpu().numpy()
        
        # Calculate bounds
        coords = np.nonzero(mask_np)
        bounds = (
            coords[0].min() if len(coords[0]) > 0 else 0,
            coords[1].min() if len(coords[0]) > 0 else 0,
            coords[0].max() if len(coords[0]) > 0 else 0,
            coords[1].max() if len(coords[0]) > 0 else 0
        )
        
        return ({
            "mask": mask_np,
            "bounds": bounds,
            "detector": detector,  # Now takes pre-configured detector
            "action": action,
            "value": value,
            "next": next_roi
        },)

class MotionDetectorNode(ControlNodeBase):
    """Configures a motion detector with specific settings"""
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()  # Get base inputs
        inputs["required"].update({
            "threshold": ("FLOAT", {
                "default": 0.1, 
                "min": 0.0, 
                "max": 1.0, 
                "step": 0.01,
                "tooltip": "Motion detection sensitivity"
            }),
            "blur_size": ("INT", {
                "default": 5, 
                "min": 1, 
                "max": 21, 
                "step": 2,
                "tooltip": "Size of blur kernel for noise reduction"
            })
        })
        return inputs

    RETURN_TYPES = ("DETECTOR",)
    FUNCTION = "update"
    CATEGORY = "real-time/control/detection"

    def update(self, threshold, blur_size, always_execute=False):
        """Implements required update method from ControlNodeBase"""
        detector = MotionDetector()
        detector.setup(threshold=threshold, blur_size=blur_size)
        return (detector,) 


class BrightnessDetectorNode(ControlNodeBase):
    """Configures a brightness detector with specific settings"""
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()  # Get base inputs
        inputs["required"].update({
            "threshold": ("FLOAT", {
                "default": 0.5, 
                "min": 0.0, 
                "max": 1.0, 
                "step": 0.01,
                "tooltip": "Brightness threshold for visualization"
            }),
            "use_average": ("BOOLEAN", {
                "default": True,
                "tooltip": "Use average brightness instead of maximum"
            })
        })
        return inputs

    RETURN_TYPES = ("DETECTOR",)
    FUNCTION = "update"
    CATEGORY = "real-time/control/detection"

    def update(self, threshold, use_average, always_execute=False):
        """Implements required update method from ControlNodeBase"""
        detector = BrightnessDetector()
        detector.setup(threshold=threshold, use_average=use_average)
        return (detector,)