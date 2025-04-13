# face_detection_mask.py
import torch
import numpy as np
import logging
import scipy.ndimage

logger = logging.getLogger(__name__)
_category = "MediaPipeVision/Face/FaceDetection"

class FaceDetectionToMask:
    """Converts face detection bounding boxes to masks that can be used in ComfyUI."""
    CATEGORY = _category
    DESCRIPTION = "Creates masks from face detections that can be used for selective processing, inpainting, or compositing. Generate masks based on face bounding boxes or facial keypoints."
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_detections": ("FACE_DETECTIONS", {"tooltip": "Face detection results from the Face Detector node"}),
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1, 
                                  "tooltip": "Width of the output mask in pixels"}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1, 
                                   "tooltip": "Height of the output mask in pixels"}),
                "mask_type": (["Bounding Box", "Keypoints", "Both"], {"default": "Bounding Box", 
                                                                     "tooltip": "Type of mask to create - Bounding Box (rectangle around face), Keypoints (dots at facial points), or Both"}),
                "face_selection": (["All Faces", "First Face", "Largest Face", "Highest Confidence Face"], 
                                  {"default": "All Faces", 
                                   "tooltip": "Which faces to include in the mask - all detected faces, only the first face, largest face, or most confident detection"}),
                "feather_amount": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, 
                                           "tooltip": "How much to soften the edges of the mask in pixels - higher values create smoother transitions"}),
                "keypoint_radius": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1, 
                                           "tooltip": "Size of each keypoint in the mask when using Keypoints or Both mask types"}),
                "invert": ("BOOLEAN", {"default": False, 
                                     "tooltip": "When enabled, inverts the mask (white becomes black, black becomes white)"}),
                "score_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                           "tooltip": "Minimum confidence score for including face detections - higher values filter out less confident detections"})
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "Optional image to match dimensions - mask will be created with the same size as this image"})
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    
    def create_mask(self, face_detections, width, height, mask_type, face_selection,
                   feather_amount, keypoint_radius, invert, score_threshold, reference_image=None):
        
        # If reference image provided, use its dimensions
        if reference_image is not None:
            height, width = reference_image.shape[1], reference_image.shape[2]
        
        batch_size = len(face_detections)
        masks = []
        
        for i in range(batch_size):
            # Create empty mask
            mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")
            detections_for_image = face_detections[i]
            
            if not detections_for_image:
                masks.append(mask)
                continue
            
            # Filter detections by score threshold
            filtered_detections = []
            for detection in detections_for_image:
                if detection.score is None or detection.score >= score_threshold:
                    filtered_detections.append(detection)
            
            # Select which faces to include based on face_selection parameter
            selected_detections = []
            if face_selection == "All Faces":
                selected_detections = filtered_detections
            elif face_selection == "First Face" and filtered_detections:
                selected_detections = [filtered_detections[0]]
            elif face_selection == "Largest Face" and filtered_detections:
                # Find the detection with the largest area
                largest_detection = max(filtered_detections, 
                                       key=lambda d: d.bounding_box.width * d.bounding_box.height)
                selected_detections = [largest_detection]
            elif face_selection == "Highest Confidence Face" and filtered_detections:
                # Find the detection with the highest confidence score
                # Filter out detections with no score first
                scored_detections = [d for d in filtered_detections if d.score is not None]
                if scored_detections:
                    highest_conf_detection = max(scored_detections, key=lambda d: d.score)
                    selected_detections = [highest_conf_detection]
                else:
                    # If no detections have scores, fall back to first detection
                    selected_detections = [filtered_detections[0]]
                
            for detection in selected_detections:
                # Get bounding box
                bbox = detection.bounding_box
                x1, y1 = max(0, bbox.origin_x), max(0, bbox.origin_y)
                x2, y2 = min(width-1, bbox.origin_x + bbox.width), min(height-1, bbox.origin_y + bbox.height)
                
                if x2 <= x1 or y2 <= y1:
                    continue  # Skip invalid boxes
                
                # Fill bounding box region if requested
                if mask_type in ["Bounding Box", "Both"]:
                    mask[y1:y2+1, x1:x2+1] = 1.0
                
                # Add keypoints if requested
                if mask_type in ["Keypoints", "Both"] and detection.keypoints:
                    for kp in detection.keypoints:
                        # Keypoints are in normalized coordinates, convert to pixel coordinates
                        cx, cy = int(kp.x * width), int(kp.y * height)
                        
                        # Draw circle for each keypoint
                        for y in range(max(0, cy - keypoint_radius), min(height, cy + keypoint_radius + 1)):
                            for x in range(max(0, cx - keypoint_radius), min(width, cx + keypoint_radius + 1)):
                                # Check if point is within radius
                                if (x - cx) ** 2 + (y - cy) ** 2 <= keypoint_radius ** 2:
                                    mask[y, x] = 1.0
            
            # Feather the edges if requested
            if feather_amount > 0:
                # Convert mask to numpy for scipy operations
                mask_np = mask.numpy()
                
                # Calculate distance transform
                if invert:
                    distance = scipy.ndimage.distance_transform_edt(mask_np == 0)
                else:
                    distance = scipy.ndimage.distance_transform_edt(mask_np > 0)
                
                # Create feathered edge
                feather_mask = np.clip(distance / feather_amount, 0, 1)
                
                if invert:
                    feather_mask = 1.0 - feather_mask
                
                # Convert back to tensor
                mask = torch.from_numpy(feather_mask).float()
            elif invert:
                # Simple inversion without feathering
                mask = 1.0 - mask
                
            masks.append(mask)
        
        # Stack all masks into a batch
        mask_batch = torch.stack(masks)
        return (mask_batch,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "FaceDetectionToMask": FaceDetectionToMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectionToMask": "Face Detection to Mask (MediaPipe)"
} 