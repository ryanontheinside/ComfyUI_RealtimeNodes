"""Node wrapper for image segmentation"""

import torch
import numpy as np
import cv2 # For visualization
import logging

# Imports from this project
from ...src.image_segmentation.segmenter import ImageSegmenter
# Import Base Classes
from ..common.model_loader import MediaPipeModelLoaderBaseNode 
from ..common.base_detector_node import BaseMediaPipeDetectorNode

logger = logging.getLogger(__name__)

_category = "MediaPipeVision/ImageSegmentation"
# Define class names and their corresponding indices for multiclass models
MULTICLASS_NAMES = {
    "Background": 0,
    "Hair": 1,
    "Body-skin": 2,
    "Face-skin": 3,
    "Clothes": 4,
    "Accessories/Other": 5,
}

# Uses MULTICLASS_NAMES indices
CLASS_COLORS_BGR = {
    MULTICLASS_NAMES["Background"]: (0, 0, 0),       # Black
    MULTICLASS_NAMES["Hair"]: (255, 0, 0),     # Blue
    MULTICLASS_NAMES["Body-skin"]: (0, 255, 255),   # Yellow
    MULTICLASS_NAMES["Face-skin"]: (255, 200, 100), # Light Blue
    MULTICLASS_NAMES["Clothes"]: (0, 255, 0),     # Green
    MULTICLASS_NAMES["Accessories/Other"]: (255, 0, 255),   # Magenta
}
# --- End Constants --- 

# Inherit from the base loader
class MediaPipeImageSegmenterModelLoaderNode(MediaPipeModelLoaderBaseNode):
    """ComfyUI node for loading MediaPipe Image Segmenter models."""
    TASK_TYPE = "image_segmenter"
    RETURN_TYPES = ("IMAGE_SEGMENTER_MODEL_INFO",)
    RETURN_NAMES = ("model_info",)
    # INPUT_TYPES and FUNCTION inherited
    CATEGORY = _category    

class MediaPipeImageSegmenterNode(BaseMediaPipeDetectorNode):
    """ComfyUI node for MediaPipe Image Segmentation, adapted from Stream-Pack.
       Manages the ImageSegmenter instance for potential reuse based on config.
    """
    
    # Define class variables required by the base class
    DETECTOR_CLASS = ImageSegmenter
    MODEL_INFO_TYPE = "IMAGE_SEGMENTER_MODEL_INFO"
    EXPECTED_TASK_TYPE = "image_segmenter"
    RETURN_TYPES = ("MASK", "IMAGE", "MP_INT_MASK",)
    RETURN_NAMES = ("mask", "visualization", "multiclass_segments")
    FUNCTION = "detect"
    CATEGORY = _category

    @classmethod
    def INPUT_TYPES(cls):
        # Start with the base inputs from the parent class
        inputs = super().INPUT_TYPES()
        
        # Add segmentation-specific parameters
        inputs["required"].update({
            "output_confidence_masks": ("BOOLEAN", {"default": False,
                                             "tooltip": "Output confidence mask (0-1) instead of category mask. Disables multiclass/visualization."}),
            "threshold": ("FLOAT", { 
                "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Confidence threshold for confidence mask output mode. Pixels below threshold become 0."
            }),
            "generate_visualization": ("BOOLEAN", {"default": False,
                "tooltip": "Generate a colored visualization image (only works in category mask mode)."
            }),
        })
        
        # Rename the delegate parameter for consistency with original implementation
        inputs["required"]["delegate_mode"] = inputs["required"].pop("delegate")
        
        return inputs

    # --- Helper methods from Stream-Pack --- 
    def is_multiclass_model(self, model_info: dict):
        # Improved check based on variant or path
        model_variant = model_info.get('model_variant', '').lower()
        model_path = model_info.get('model_path', '').lower()
        # Add known multiclass identifiers
        return 'multiclass' in model_variant or 'multiclass' in model_path

    def create_visualization(self, category_mask_np, is_multiclass=False):
        """Generates a colored visualization from a category mask."""
        if category_mask_np is None:
             return None
        if not is_multiclass:
            # Simple binary visualization (white on black)
            vis_image = np.where(category_mask_np > 0, 255, 0).astype(np.uint8)
            return cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        else:
            # Multiclass visualization
            height, width = category_mask_np.shape
            vis_image = np.zeros((height, width, 3), dtype=np.uint8)
            for class_name, class_id in MULTICLASS_NAMES.items():
                 color_bgr = CLASS_COLORS_BGR.get(class_id, (128, 128, 128)) # Default grey
                 vis_image[category_mask_np == class_id] = color_bgr
            return vis_image
    # --- End Helper methods --- 

    def detect(self, image: torch.Tensor, model_info: dict, output_confidence_masks: bool, 
               threshold: float, generate_visualization: bool, running_mode: str, delegate_mode: str):
        """Performs image segmentation with the configured parameters."""
        
        # Validate model_info and get model path
        model_path = self.validate_model_info(model_info)
        
        # Initialize or update detector
        detector = self.initialize_or_update_detector(model_path)
        
        # Determine which output type the detector needs
        request_confidence = output_confidence_masks
        # Need category mask if confidence isn't requested OR if visualization is needed
        request_category = not output_confidence_masks or generate_visualization 
        
        if not request_confidence and not request_category:
             # Should not happen with the logic above, but as a safeguard
             raise ValueError("Internal logic error: No mask type determined for segmenter.")

        # Call segmenter's segment method
        # Returns lists (one element per image in batch)
        batch_results_confidence, batch_results_category = detector.segment(
            image,
            output_confidence_masks=request_confidence,
            output_category_mask=request_category, 
            running_mode=running_mode,
            delegate_mode=delegate_mode
        )

        # Initialize output lists
        batch_size = image.shape[0]
        h, w = image.shape[1], image.shape[2]
        all_primary_masks = []
        all_vis_images = []
        all_category_masks = []
        default_vis_tensor = torch.zeros((h, w, 3), dtype=torch.float32, device=image.device) # HWC

        # --- Process results based on node inputs --- 
        is_multiclass = self.is_multiclass_model(model_info)
        
        # Loop through batch results
        for i in range(batch_size):
            # Get results for the current image
            confidence_masks_hw_list = batch_results_confidence[i] if batch_results_confidence else None
            category_mask_hw = batch_results_category[i] if batch_results_category else None
            
            # Initialize outputs for this image
            current_primary_mask_hw = torch.zeros((h, w), dtype=torch.float32, device=image.device)
            current_vis_image_hwc = default_vis_tensor
            current_category_mask_hw = torch.zeros((h, w), dtype=torch.long, device=image.device)

            if output_confidence_masks:
                # User wants confidence mask as primary output
                if confidence_masks_hw_list:
                    # Take the first confidence mask (often foreground vs background)
                    conf_mask_hw = confidence_masks_hw_list[0] 
                    # Apply threshold
                    conf_mask_hw_thresh = torch.where(conf_mask_hw >= threshold, conf_mask_hw, torch.zeros_like(conf_mask_hw))
                    current_primary_mask_hw = conf_mask_hw_thresh # HW
                # Visualization and multiclass output are disabled/invalid in confidence mode
                generate_visualization_for_this = False 
            else: 
                # User wants category mask (default or for visualization)
                if category_mask_hw is not None:
                     # Determine primary MASK output (HW Float)
                     if is_multiclass: 
                         # Treat non-background as foreground for the primary mask
                         primary_mask_hw = (category_mask_hw > 0).float() 
                     else:
                         # Binary models: Category 1 is foreground
                         primary_mask_hw = (category_mask_hw > 0).float()
                     current_primary_mask_hw = primary_mask_hw # HW
                     
                     # Prepare raw category mask output (HW Long Tensor)
                     current_category_mask_hw = category_mask_hw # Already Long
                     
                     # Generate visualization if requested (and category mask exists)
                     generate_visualization_for_this = generate_visualization
                     if generate_visualization_for_this:
                         # Convert to NumPy for OpenCV, then back to tensor
                         category_np = category_mask_hw.cpu().numpy().astype(np.uint8)
                         vis_np = self.create_visualization(category_np, is_multiclass)
                         if vis_np is not None:
                             # Convert back to tensor (HWC)
                             vis_tensor = torch.from_numpy(vis_np).to(dtype=torch.float32, device=image.device) / 255.0
                             current_vis_image_hwc = vis_tensor # HWC
            
            # Add results for this image to the batch results
            all_primary_masks.append(current_primary_mask_hw) # HW
            all_vis_images.append(current_vis_image_hwc) # HWC
            all_category_masks.append(current_category_mask_hw) # HW
            
        # Stack all results to return as tensors
        primary_mask_tensor = torch.stack(all_primary_masks, dim=0) # BHW
        vis_image_tensor = torch.stack(all_vis_images, dim=0) # BHWC
        category_mask_tensor = torch.stack(all_category_masks, dim=0) # BHW
        
        # Ensure multiclass_segments is in the right format (BHW) for downstream nodes
        # Not adding an unsqueeze here, as we're handling both formats in the select_segment node
        
        return (primary_mask_tensor, vis_image_tensor, category_mask_tensor)

# Define mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "MediaPipeImageSegmenterModelLoader": MediaPipeImageSegmenterModelLoaderNode,
    "MediaPipeImageSegmenter": MediaPipeImageSegmenterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeImageSegmenterModelLoader": "Load Image Segmenter Model (MediaPipe)",
    "MediaPipeImageSegmenter": "Image Segmenter (MediaPipe)",
} 