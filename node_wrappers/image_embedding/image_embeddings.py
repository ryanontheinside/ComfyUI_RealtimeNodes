import torch
import logging

# Import Base Loader and Detector
from ..common.model_loader import MediaPipeModelLoaderBaseNode
from ..common.base_detector_node import BaseMediaPipeDetectorNode
from ...src.image_embedding.detector import ImageEmbedder

logger = logging.getLogger(__name__)
_category = "MediaPipeVision/ImageEmbedding"

# --- Model Loader --- 
class MediaPipeImageEmbedderModelLoaderNode(MediaPipeModelLoaderBaseNode):
    """ComfyUI node for loading MediaPipe Image Embedder models."""
    TASK_TYPE = "image_embedder" # Need to add this task to MEDIAPIPE_MODELS
    RETURN_TYPES = ("IMAGE_EMBEDDER_MODEL_INFO",)
    RETURN_NAMES = ("model_info",)
    CATEGORY = _category    

# --- Embedder Node --- 
class MediaPipeImageEmbedderNode(BaseMediaPipeDetectorNode):
    """ComfyUI node for MediaPipe Image Embedding."""

    # Define class variables required by the base class
    DETECTOR_CLASS = ImageEmbedder
    MODEL_INFO_TYPE = "IMAGE_EMBEDDER_MODEL_INFO"
    EXPECTED_TASK_TYPE = "image_embedder"
    RETURN_TYPES = ("IMAGE_EMBEDDINGS",)
    RETURN_NAMES = ("image_embeddings",)
    FUNCTION = "detect"
    CATEGORY = _category

    @classmethod
    def INPUT_TYPES(cls):
        # Start with the base inputs from the parent class
        inputs = super().INPUT_TYPES()
        
        # Add embedder-specific parameters
        inputs["required"].update({
            "l2_normalize": ("BOOLEAN", {"default": True, 
                                      "tooltip": "Apply L2 normalization to the embedding"}),
            "quantize": ("BOOLEAN", {"default": False, 
                                    "tooltip": "Output quantized (uint8) embedding instead of float"}),
        })
        
        return inputs

    def detect(self, image: torch.Tensor, model_info: dict, l2_normalize: bool, 
               quantize: bool, running_mode: str, delegate: str):
        """Generates image embeddings with the configured parameters."""
        
        # Validate model_info and get model path
        model_path = self.validate_model_info(model_info)
        
        # Initialize or update detector
        detector = self.initialize_or_update_detector(model_path)
        
        # Generate embeddings with all parameters
        embeddings_batch = detector.embed(
            image,
            l2_normalize=l2_normalize,
            quantize=quantize,
            running_mode=running_mode,
            delegate=delegate
        )
        
        return (embeddings_batch,)

# --- Mappings --- 
NODE_CLASS_MAPPINGS = {
    "MediaPipeImageEmbedderModelLoader": MediaPipeImageEmbedderModelLoaderNode,
    "MediaPipeImageEmbedder": MediaPipeImageEmbedderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MediaPipeImageEmbedderModelLoader": "Load Image Embedder Model (MediaPipe)",
    "MediaPipeImageEmbedder": "Image Embedder (MediaPipe)",
} 