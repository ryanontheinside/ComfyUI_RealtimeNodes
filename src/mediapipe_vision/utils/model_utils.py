"""Utilities for managing MediaPipe Vision models."""

import os
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ComfyUI's folder_paths to use the models directory
try:
    import folder_paths
    HAS_COMFY_PATHS = True
except ImportError:
    logger.warning("ComfyUI folder_paths module not available. Using local paths.")
    HAS_COMFY_PATHS = False


def get_mediapipe_models_directory() -> str:
    """Get the directory where MediaPipe models are stored.
    
    Uses ComfyUI's models directory structure when available.
    
    Returns:
        Path to the MediaPipe models directory
    """
    if HAS_COMFY_PATHS:
        base_path = folder_paths.models_dir
        mediapipe_dir = os.path.join(base_path, "mediapipe", "vision")
    else:
        # Fallback to local path
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        mediapipe_dir = os.path.join(base_path, "models")
    
    # Create the directory if it doesn't exist
    os.makedirs(mediapipe_dir, exist_ok=True)
    return mediapipe_dir


def get_model_path(model_type: str, model_name: str) -> str:
    """Get the path for a specific MediaPipe Vision model.
    
    Args:
        model_type: Type/category of model (e.g., 'face_mesh', 'pose')
        model_name: Name of the specific model file
        
    Returns:
        Full path to the model file
    """
    models_dir = get_mediapipe_models_directory()
    model_type_dir = os.path.join(models_dir, model_type)
    os.makedirs(model_type_dir, exist_ok=True)
    
    return os.path.join(model_type_dir, model_name)


# Model information dictionary with URLs and files for each model type
# Using a nested dictionary structure to organize by task
MEDIAPIPE_MODELS = {
    "face_detection": {
        "short_range": {
            "url": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.task",
            "filename": "blaze_face_short_range.task"
        },
        "full_range": {
            "url": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face/float16/1/blaze_face.task",
            "filename": "blaze_face.task"
        }
    },
    "face_mesh": {
        "standard": {
            "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            "filename": "face_landmarker.task",
            "description": "Detects face landmarks and facial features (468 points)."
        },
        "attention": {
            "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/2/face_landmarker.task",
            "filename": "face_landmarker_v2.task",
            "description": "Improved version with better accuracy for face landmarks."
        }
    },
    "hand_tracking": {
        "standard": {
            "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            "filename": "hand_landmarker.task",
            "description": "Detects hand landmarks (21 points per hand)."
        }
    },
    "pose_detection": {
        "lite": {
            "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            "filename": "pose_landmarker_lite.task",
            "description": "Lightweight pose detection model (33 points)."
        },
        "full": {
            "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
            "filename": "pose_landmarker_full.task",
            "description": "Full pose detection model with high accuracy."
        },
        "heavy": {
            "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
            "filename": "pose_landmarker_heavy.task",
            "description": "Most accurate pose detection model."
        }
    },
    "image_segmentation": {
        "selfie": {
            "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.task",
            "filename": "selfie_segmenter.task",
            "description": "Segments people from background."
        },
        "selfie_landscape": {
            "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.task",
            "filename": "selfie_segmenter_landscape.task",
            "description": "Optimized for landscape orientation."
        },
        "multiclass": {
            "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.task",
            "filename": "selfie_multiclass_256x256.task",
            "description": "Segments multiple body parts (face, hair, clothes, etc.)."
        }
    },
    "object_detection": {
        "efficientdet_lite0": {
            "url": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.task",
            "filename": "efficientdet_lite0.task",
            "description": "General object detection (80 classes)."
        },
        "efficientdet_lite2": {
            "url": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/1/efficientdet_lite2.task",
            "filename": "efficientdet_lite2.task",
            "description": "Higher accuracy object detection."
        }
    },
    "image_classification": {
        "efficientnet_lite0": {
            "url": "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.task",
            "filename": "efficientnet_lite0.task",
            "description": "General image classification (1000 classes)."
        },
        "efficientnet_lite4": {
            "url": "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite4/float32/2/efficientnet_lite4.task",
            "filename": "efficientnet_lite4.task",
            "description": "Higher accuracy image classification."
        }
    }
} 