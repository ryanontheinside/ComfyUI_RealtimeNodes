from ..base.control_base import ControlNodeBase
from enum import Enum
import numpy as np
import torch
import cv2
import time

class ROIAction(Enum):
    # Behavioral actions
    TOGGLE = "toggle"      # Toggles between min/max values
    MOMENTARY = "momentary"  # Outputs max while motion detected
    TRIGGER = "trigger"    # Triggers once per motion event
    COUNTER = "counter"    # Counts motion events
    # Mathematical actions
    ADD = "add"          # Add value when motion detected
    SUBTRACT = "subtract" # Subtract value when motion detected
    MULTIPLY = "multiply" # Multiply by value when motion detected
    DIVIDE = "divide"     # Divide by value when motion detected
    SET = "set"          # Set to specific value when motion detected

class ROINode(ControlNodeBase):
    """Defines a single region of interest and its action"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),  # ROI defined by mask
                "action": (list(action.value for action in ROIAction),),
                "value": ("FLOAT", {
                    "default": 0.1,
                    "tooltip": "Value to use for mathematical operations"
                }),
            },
            "optional": {
                "next_roi": ("ROI",)  # Chain to next ROI
            }
        }

    RETURN_TYPES = ("ROI",)
    FUNCTION = "define_roi"
    CATEGORY = "real-time/control/motion"

    def define_roi(self, mask, action, value, next_roi=None):
        # Convert mask to numpy for bounding box calculation
        mask_np = mask[0].cpu().numpy()
        
        # Find non-zero coordinates in mask
        coords = np.nonzero(mask_np)
        if len(coords[0]) == 0:  # Empty mask
            y_min = x_min = y_max = x_max = 0
        else:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
        
        roi_data = {
            "mask": mask_np,
            "bounds": (y_min, x_min, y_max, x_max),
            "action": action,
            "value": value,  # Store the operation value
            "next": next_roi
        }
        return (roi_data,)

class MotionController(ControlNodeBase):
    """Processes motion detection for a chain of ROIs and manages their states"""
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "image": ("IMAGE",),
            "roi_chain": ("ROI",),
            "threshold": ("FLOAT", {
                "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Motion detection threshold"
            }),
            "blur_size": ("INT", {
                "default": 5, "min": 1, "max": 21, "step": 2,
                "tooltip": "Size of Gaussian blur kernel"
            }),
            "minimum_value": ("FLOAT", {
                "default": 0.0,
                "tooltip": "Minimum output value"
            }),
            "maximum_value": ("FLOAT", {
                "default": 1.0,
                "tooltip": "Maximum output value"
            }),
            "starting_value": ("FLOAT", {
                "default": 0.0,
                "tooltip": "Initial output value"
            })
        })
        return inputs

    RETURN_TYPES = ("FLOAT", "MASK")
    FUNCTION = "process_motion"
    CATEGORY = "real-time/control/motion"

    def process_motion(self, image, roi_chain, threshold, blur_size, 
                      minimum_value, maximum_value, starting_value, always_execute=True):
        # Get or initialize state with new caching fields
        state = self.get_state({
            "prev_frame": None,
            "prev_frame_blurred": None,  # Cache blurred previous frame
            "roi_states": {},
            "current_value": starting_value,
            "last_cleanup": time.time()  # Track last state cleanup
        })
        
        # Convert image tensor to numpy array and prepare current frame
        current_frame = (image[0] * 255).cpu().numpy().astype(np.uint8)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        current_blurred = cv2.GaussianBlur(current_gray, (blur_size, blur_size), 0)
        
        # Initialize motion mask
        motion_mask = np.zeros_like(current_gray, dtype=np.float32)
        
        if state["prev_frame"] is not None:
            # Collect active ROIs for state cleanup
            active_rois = set()
            
            # Process each ROI in the chain
            current_roi = roi_chain
            while current_roi is not None:
                mask = current_roi["mask"]
                bounds = current_roi["bounds"]
                action = current_roi["action"]
                roi_id = str(bounds)
                active_rois.add(roi_id)
                
                # Extract ROI regions once
                y_min, x_min, y_max, x_max = bounds
                roi_region = slice(y_min, y_max+1), slice(x_min, x_max+1)
                
                # Process ROI using cached blurred frames
                roi_current = current_blurred[roi_region]
                roi_prev = state["prev_frame_blurred"][roi_region]
                roi_mask = mask[roi_region]
                
                # Detect motion in ROI
                diff = cv2.absdiff(roi_current, roi_prev)
                _, thresh = cv2.threshold(diff, threshold * 255, 255, cv2.THRESH_BINARY)
                thresh = thresh * (roi_mask > 0.5)
                
                # Update motion mask
                motion_mask[roi_region] = np.maximum(
                    motion_mask[roi_region],
                    thresh / 255.0
                )
                
                # Calculate motion magnitude
                roi_area = np.sum(roi_mask > 0.5)
                motion_detected = np.sum(thresh) / (255 * roi_area) > threshold if roi_area > 0 else False
                
                # Get or initialize ROI state
                roi_state = state["roi_states"].get(roi_id, {"active": False, "count": 0})
                value = current_roi["value"]
                
                # Process actions with optimized value updates
                if motion_detected and not roi_state["active"]:
                    roi_state["active"] = True
                    
                    # Update current value based on action type
                    if action == ROIAction.ADD.value:
                        state["current_value"] = min(maximum_value, state["current_value"] + value)
                    elif action == ROIAction.SUBTRACT.value:
                        state["current_value"] = max(minimum_value, state["current_value"] - value)
                    elif action == ROIAction.MULTIPLY.value:
                        state["current_value"] = min(maximum_value, state["current_value"] * value)
                    elif action == ROIAction.DIVIDE.value:
                        if value != 0:
                            state["current_value"] = max(minimum_value, state["current_value"] / value)
                    elif action == ROIAction.SET.value:
                        state["current_value"] = max(minimum_value, min(maximum_value, value))
                    elif action == ROIAction.TOGGLE.value:
                        state["current_value"] = maximum_value if state["current_value"] == minimum_value else minimum_value
                    elif action == ROIAction.TRIGGER.value:
                        state["current_value"] = maximum_value
                    elif action == ROIAction.COUNTER.value:
                        roi_state["count"] += 1
                        state["current_value"] = minimum_value + (
                            roi_state["count"] % (int((maximum_value - minimum_value) + 1))
                        )
                
                elif action == ROIAction.MOMENTARY.value:
                    state["current_value"] = maximum_value if motion_detected else minimum_value
                
                elif not motion_detected:
                    roi_state["active"] = False
                    if action == ROIAction.TRIGGER.value:
                        state["current_value"] = minimum_value
                
                state["roi_states"][roi_id] = roi_state
                current_roi = current_roi["next"]
            
            # Cleanup inactive ROI states periodically (every 60 seconds)
            if time.time() - state["last_cleanup"] > 60:
                state["roi_states"] = {k: v for k, v in state["roi_states"].items() 
                                     if k in active_rois}
                state["last_cleanup"] = time.time()
        
        # Update state with current frame data
        state["prev_frame"] = current_gray
        state["prev_frame_blurred"] = current_blurred
        self.set_state(state)
        
        # Convert mask to tensor
        mask_tensor = torch.from_numpy(motion_mask).unsqueeze(0)
        
        return (state["current_value"], mask_tensor) 