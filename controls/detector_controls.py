import sys
import os
import time
import numpy as np
import torch

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from base.control_base import ControlNodeBase
from base.detector_base import ROIAction, SharedProcessing

class DetectionControlBase(ControlNodeBase):
    """Base class for all detection-based controllers"""
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()  # Get base inputs including always_execute
        inputs["required"].update({
            "image": ("IMAGE",),
            "roi_chain": ("ROI",),
        })
        return inputs

    RETURN_TYPES = ("MASK",)
    CATEGORY = "real-time/control/detection"

    def update(self, *args, **kwargs):
        """Implement abstract method from ControlNodeBase"""
        return self.process_detection_base(*args, **kwargs)

    def _process_action(self, state, action, detection, value, roi_state, min_val, max_val):
        """Process ROI action based on detection"""
        detected = detection > 0.5  # Basic threshold
        
        if detected and not roi_state["active"]:
            roi_state["active"] = True
            
            if action == ROIAction.ADD.value:
                state["current_value"] = min(max_val, state["current_value"] + value)
            elif action == ROIAction.SUBTRACT.value:
                state["current_value"] = max(min_val, state["current_value"] - value)
            elif action == ROIAction.MULTIPLY.value:
                state["current_value"] = min(max_val, state["current_value"] * value)
            elif action == ROIAction.DIVIDE.value:
                if value != 0:
                    state["current_value"] = max(min_val, state["current_value"] / value)
            elif action == ROIAction.SET.value:
                state["current_value"] = max(min_val, min(max_val, value))
            elif action == ROIAction.TOGGLE.value:
                state["current_value"] = max_val if state["current_value"] == min_val else min_val
            elif action == ROIAction.TRIGGER.value:
                state["current_value"] = max_val
            elif action == ROIAction.COUNTER.value:
                roi_state["count"] += 1
                state["current_value"] = min_val + (
                    roi_state["count"] % (int((max_val - min_val) + 1))
                )
        
        elif action == ROIAction.MOMENTARY.value:
            state["current_value"] = max_val if detected else min_val
        
        elif not detected:
            roi_state["active"] = False
            if action == ROIAction.TRIGGER.value:
                state["current_value"] = min_val

    def process_detection_base(self, image, roi_chain, minimum_value, maximum_value, starting_value, always_execute=True):
        state = self.get_state({
            "current_value": starting_value,
            "detector_states": {},
            "roi_states": {},
            "last_cleanup": time.time()
        })
        
        # Convert image tensor to numpy
        current_frame = (image[0] * 255).cpu().numpy().astype(np.uint8)
        detection_mask = np.zeros_like(current_frame[:,:,0], dtype=np.float32)

        # Generate shared preprocessed data once
        shared_data = SharedProcessing.get_shared_data(current_frame)
        
        # Collect all unique detector types from ROI chain
        detector_types = set()
        current_roi = roi_chain
        while current_roi is not None:
            detector = current_roi["detector"]
            detector_types.add(detector.__class__)
            current_roi = current_roi["next"]
        
        # Run preprocessing for each detector type
        preprocessed_data = {}
        for detector_type in detector_types:
            if hasattr(detector_type, 'preprocess'):
                preprocessed_data[detector_type] = detector_type.preprocess(current_frame, shared_data)
        
        # Process ROI chain
        current_roi = roi_chain
        while current_roi is not None:
            bounds = current_roi["bounds"]
            y_min, x_min, y_max, x_max = bounds
            roi_id = str(bounds)
            
            # Get detector state
            detector_state = state["detector_states"].setdefault(roi_id, {})
            detector_state.update({
                'y_offset': y_min,
                'x_offset': x_min,
                'shared': shared_data  # Add shared data
            })
            
            # Add preprocessed data if available
            detector_type = current_roi["detector"].__class__
            if detector_type in preprocessed_data:
                detector_state["preprocessed"] = preprocessed_data[detector_type]
            
            # Get ROI state
            roi_state = state["roi_states"].setdefault(roi_id, {
                "active": False,
                "count": 0
            })
            
            # Run detection
            detection_value, viz_mask = current_roi["detector"].detect(
                current_frame[y_min:y_max+1, x_min:x_max+1],
                current_roi["mask"][y_min:y_max+1, x_min:x_max+1],
                detector_state
            )
            
            # Update visualization mask
            detection_mask[y_min:y_max+1, x_min:x_max+1] = np.maximum(
                detection_mask[y_min:y_max+1, x_min:x_max+1],
                viz_mask
            )
            
            # Process action
            self._process_action(
                state, current_roi["action"],
                detection_value, current_roi["value"],
                roi_state, minimum_value, maximum_value
            )
            
            current_roi = current_roi["next"]
        
        self.set_state(state)
        return state["current_value"], torch.from_numpy(detection_mask).unsqueeze(0)

class FloatDetectionControl(DetectionControlBase):
    """Controls a floating point value based on detection events"""
    
    DESCRIPTION = "Generates a floating point value based on detection events in regions of interest"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "maximum_value": ("FLOAT", {
                "default": 1.0,
                "min": -10000.0,
                "max": 10000.0,
                "step": 0.1,
                "tooltip": "Maximum value that can be output"
            }),
            "minimum_value": ("FLOAT", {
                "default": 0.0,
                "min": -10000.0,
                "max": 10000.0,
                "step": 0.1,
                "tooltip": "Minimum value that can be output"
            }),
            "starting_value": ("FLOAT", {
                "default": 0.5,
                "min": -10000.0,
                "max": 10000.0,
                "step": 0.1,
                "tooltip": "Initial value when the node first executes"
            })
        })
        return inputs

    RETURN_TYPES = ("FLOAT", "MASK")
    FUNCTION = "process_detection"
    
    def process_detection(self, *args, **kwargs):
        value, mask = self.process_detection_base(*args, **kwargs)
        return (float(value), mask)

class IntDetectionControl(DetectionControlBase):
    """Controls an integer value based on detection events"""
    
    DESCRIPTION = "Generates an integer value based on detection events in regions of interest"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "maximum_value": ("INT", {
                "default": 100,
                "min": -10000,
                "max": 10000,
                "step": 1,
                "tooltip": "Maximum value that can be output"
            }),
            "minimum_value": ("INT", {
                "default": 0,
                "min": -10000,
                "max": 10000,
                "step": 1,
                "tooltip": "Minimum value that can be output"
            }),
            "starting_value": ("INT", {
                "default": 50,
                "min": -10000,
                "max": 10000,
                "step": 1,
                "tooltip": "Initial value when the node first executes"
            })
        })
        return inputs

    RETURN_TYPES = ("INT", "MASK")
    FUNCTION = "process_detection"
    
    def process_detection(self, *args, **kwargs):
        value, mask = self.process_detection_base(*args, **kwargs)
        return (int(round(value)), mask)

class StringDetectionControl(DetectionControlBase):
    """Controls string output based on detection events"""
    
    DESCRIPTION = "Outputs strings based on detection events in regions of interest"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "strings": ("STRING", {
                "multiline": True,
                "default": "first string\nsecond string\nthird string",
                "tooltip": "List of strings to cycle through (one per line)"
            })
        })
        return inputs

    RETURN_TYPES = ("STRING", "MASK")
    FUNCTION = "process_detection"
    
    def process_detection(self, image, roi_chain, strings, always_execute=True):
        # Split strings into list
        string_list = [s.strip() for s in strings.split('\n') if s.strip()]
        if not string_list:
            return ("", torch.zeros((1, image.shape[1], image.shape[2])))
            
        # Process with normalized range
        value, mask = self.process_detection_base(
            image, roi_chain,
            minimum_value=0,
            maximum_value=len(string_list) - 1,
            starting_value=0,
            always_execute=always_execute
        )
        
        # Convert value to string index
        index = int(round(value)) % len(string_list)
        return (string_list[index], mask)

    def _process_action(self, state, action, detection, value, roi_state, min_val, max_val):
        """Process ROI action based on detection"""
        detected = detection > 0.5  # Basic threshold
        
        if detected and not roi_state["active"]:
            roi_state["active"] = True
            
            if action == ROIAction.ADD.value:
                state["current_value"] = min(max_val, state["current_value"] + value)
            elif action == ROIAction.SUBTRACT.value:
                state["current_value"] = max(min_val, state["current_value"] - value)
            elif action == ROIAction.MULTIPLY.value:
                state["current_value"] = min(max_val, state["current_value"] * value)
            elif action == ROIAction.DIVIDE.value:
                if value != 0:
                    state["current_value"] = max(min_val, state["current_value"] / value)
            elif action == ROIAction.SET.value:
                state["current_value"] = max(min_val, min(max_val, value))
            elif action == ROIAction.TOGGLE.value:
                state["current_value"] = max_val if state["current_value"] == min_val else min_val
            elif action == ROIAction.TRIGGER.value:
                state["current_value"] = max_val
            elif action == ROIAction.COUNTER.value:
                roi_state["count"] += 1
                state["current_value"] = min_val + (
                    roi_state["count"] % (int((max_val - min_val) + 1))
                )
        
        elif action == ROIAction.MOMENTARY.value:
            state["current_value"] = max_val if detected else min_val
        
        elif not detected:
            roi_state["active"] = False
            if action == ROIAction.TRIGGER.value:
                state["current_value"] = min_val 