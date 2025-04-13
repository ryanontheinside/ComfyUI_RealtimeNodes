"""Model loader implementation for MediaPipe Vision tasks.

Handles model definition, download, and path retrieval.
"""

import os
import logging
import urllib.request
import folder_paths # ComfyUI path helper
from typing import Dict, Any, Optional, List
import traceback

# Configure logging
# logging.basicConfig(level=logging.INFO) # Avoid reconfiguring root logger
logger = logging.getLogger(__name__)

# --- Model Definitions --- 
# Central dictionary mapping task types and variants to model details
# Structure: { task_type: { variant_name: { url: ..., filename: ... } } }
# Task types should match the keys used elsewhere (e.g., detector class validation)
# Variant names are displayed in the loader node UI
# Filenames should be unique within their task type directory
# URLs point to the official MediaPipe model files (.task or .tflite)

MEDIAPIPE_MODELS = {
    "face_detector": {
        "short_range": {
            "url": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
            "filename": "blaze_face_short_range.tflite",
            "description": "Optimized for detecting faces at close range, ideal for selfie-style images and near-field face detection."
        },
        "full_range_sparse": {
            "url": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_back_camera/float16/latest/blaze_face_back_camera.tflite",
            "filename": "blaze_face_full_range_sparse.tflite",
            "description": "Designed for detecting faces at various distances, suitable for group photos and wider-angle shots."
        }
    },
    "face_landmarker": {
        "default": {
            "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            "filename": "face_landmarker.task",
            "description": "Standard face landmark detection model that locates key facial points for tracking and analysis."
        },
         "with_blendshapes": { # If a separate model exists
            "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker_v2_with_blendshapes/float16/latest/face_landmarker_v2_with_blendshapes.task",
            "filename": "face_landmarker_v2_with_blendshapes.task",
            "description": "Advanced face landmark model that includes blendshape coefficients for facial expression analysis."
         }
    },
    "hand_landmarker": {
        "default": { # Often includes world landmarks
            "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            "filename": "hand_landmarker.task",
            "description": "Detects hand landmarks for tracking finger positions and hand gestures in 2D and 3D space."
        }
    },
    "pose_landmarker": {
        "lite": {
            "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
            "filename": "pose_landmarker_lite.task",
            "description": "Lightweight pose landmark model for faster inference with slightly reduced accuracy."
        },
        "full": {
            "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
            "filename": "pose_landmarker_full.task",
            "description": "Balance of performance and accuracy for detecting body pose landmarks in typical use cases."
        },
        "heavy": {
            "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
            "filename": "pose_landmarker_heavy.task",
            "description": "Higher accuracy pose detection model for more precise landmark tracking at the cost of performance."
        }
    },
    "object_detector": {
        "efficientdet_lite0": {
            "url": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite",
            "filename": "efficientdet_lite0.tflite",
            "description": "Lightweight object detection model that balances speed and accuracy for real-time applications."
        },
        "efficientdet_lite2": {
             "url": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/latest/efficientdet_lite2.tflite",
             "filename": "efficientdet_lite2.tflite",
             "description": "Medium-sized object detection model with improved accuracy over lite0 while maintaining reasonable performance."
         },
         "ssd_mobilenet_v2": {
             "url": "https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float32/latest/ssd_mobilenet_v2.tflite",
             "filename": "ssd_mobilenet_v2.tflite",
             "description": "Single Shot Detector with MobileNetV2 backbone for efficient object detection tasks."
         }
    },
    "image_segmenter": {
        "deeplab_v3": {
            "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/latest/deeplab_v3.tflite",
            "filename": "deeplab_v3.tflite",
            "description": "General-purpose semantic segmentation model that can segment images into different object categories."
        },
        "selfie_segmenter_landscape": {
            "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
            "filename": "selfie_segmenter_landscape.tflite",
            "description": "Optimized for landscape-oriented images to accurately separate people from backgrounds."
        },
        "selfie_segmenter_general": {
            "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
            "filename": "selfie_segmenter.tflite",
            "description": "General-purpose person segmentation model that works across different orientations and scenarios."
        },
        "hair_segmenter": {
            "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite",
            "filename": "hair_segmenter.tflite",
            "description": "Specialized segmentation model for isolating hair in portrait images with high precision."
        },
        "selfie_multiclass": {
            "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
            "filename": "selfie_multiclass_256x256.tflite",
            "description": "Multi-class segmentation model that can separate people into different body parts and clothing items."
        }
    },
    "gesture_recognizer": {
        "default": {
            "url": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
            "filename": "gesture_recognizer.task",
            "description": "Detects and classifies hand gestures such as thumbs up, victory sign, and pointing gestures."
        }
    },
    # "holistic_landmarker": {
    #     "lite": {
    #         "url": "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker_lite/holistic_landmarker_lite.task",
    #         "filename": "holistic_landmarker_lite.task"
    #     },
    #      "full": {
    #         "url": "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker_full/holistic_landmarker_full.task",
    #         "filename": "holistic_landmarker_full.task"
    #     }
    # }
    # Add other tasks like image_classification etc. here
    "image_embedder": {
        "mobilenet_v3_small": { 
            "url": "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite",
            "filename": "mobilenet_v3_small.tflite",
            "description": "Converts images into feature vectors (embeddings) for similarity comparison and classification tasks."
        },
        "efficientnet_lite0": {
            "url": "https://storage.googleapis.com/mediapipe-models/image_embedder/efficientnet_lite0/float32/latest/efficientnet_lite0.tflite",
            "filename": "efficientnet_lite0.tflite",
            "description": "EfficientNet Lite0-based embedder providing robust image embeddings with a balance of accuracy and efficiency."
        }
    },
    # Placeholder for Interactive Segmenter - Typically uses a general segmentation model
    "interactive_segmenter": {
        "magic_touch": {
            "url": "https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/latest/magic_touch.tflite",
            "filename": "magic_touch.tflite",
            "description": "Interactive segmentation model that segments objects in an image based on a user-provided point of interest."
        },
        "default": {
             "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
             "filename": "selfie_segmenter_interactive.tflite",
             "description": "Enables user-guided segmentation by providing points or regions to indicate foreground objects."
        }
    },
    "face_stylizer": {
        "color_sketch": {
            "url": "https://storage.googleapis.com/mediapipe-models/face_stylizer/blaze_face_stylizer/float32/latest/face_stylizer_color_sketch.task",
            "filename": "face_stylizer_color_sketch.task",
            "description": "Transforms faces into colorful sketch-like artistic renderings with clean lines and vibrant colors."
        },
        "color_ink": {
            "url": "https://storage.googleapis.com/mediapipe-models/face_stylizer/blaze_face_stylizer/float32/latest/face_stylizer_color_ink.task",
            "filename": "face_stylizer_color_ink.task",
            "description": "Applies a color ink effect to faces with flowing brush-like strokes and dynamic color blending."
        },
        "oil_painting": {
            "url": "https://storage.googleapis.com/mediapipe-models/face_stylizer/blaze_face_stylizer/float32/latest/face_stylizer_oil_painting.task",
            "filename": "face_stylizer_oil_painting.task",
            "description": "Renders faces in classic oil painting style with textured brush strokes and rich color tones."
        }
    },
    "image_classifier": {
        "efficientnet_lite0": {
            "url": "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/latest/efficientnet_lite0.tflite",
            "filename": "efficientnet_lite0.tflite",
            "description": "EfficientNet-Lite0 model for general-purpose image classification tasks with optimized performance."
        },
        "efficientnet_lite2": {
            "url": "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite2/float32/latest/efficientnet_lite2.tflite",
            "filename": "efficientnet_lite2.tflite",
            "description": "EfficientNet-Lite2 model offering higher accuracy for image classification tasks at the cost of increased computation."
        }
    },
    "face_mesh": {
        "default": {
            "url": "https://storage.googleapis.com/mediapipe-models/face_mesh/face_mesh/float16/latest/face_mesh.tflite",
            "filename": "face_mesh.tflite",
            "description": "High-fidelity 3D face landmark model estimating 468 landmarks for detailed facial geometry analysis."
        },
        "attention": {
            "url": "https://storage.googleapis.com/mediapipe-models/face_mesh/face_mesh_attention/float16/latest/face_mesh_attention.tflite",
            "filename": "face_mesh_attention.tflite",
            "description": "Enhanced face mesh model with attention mechanism for improved accuracy around eyes and lips."
        }
    }
}

# --- Utility Functions --- 

def get_mediapipe_model_dir(task_type: str) -> str:
    """Gets the dedicated directory for a specific MediaPipe task type within ComfyUI models."""
    base_path = None
    try:
        # Attempt to get the specific 'mediapipe' path defined in ComfyUI config
        mediapipe_paths = folder_paths.get_folder_paths("mediapipe")
        if mediapipe_paths and os.path.isdir(mediapipe_paths[0]):
            base_path = mediapipe_paths[0]
            logger.debug(f"Using specific MediaPipe model path: {base_path}")
        else:
             logger.debug("No valid specific 'mediapipe' path found or defined.")
    except (KeyError, IndexError, TypeError) as e:
        logger.debug(f"Could not retrieve specific 'mediapipe' path ({e}). Falling back to default.")

    # If specific path wasn't found or valid, use the main models directory
    if base_path is None:
        try:
            # Ensure models_dir exists
            if hasattr(folder_paths, 'models_dir') and os.path.isdir(folder_paths.models_dir):
                base_path = os.path.join(folder_paths.models_dir, "mediapipe")
                logger.debug(f"Using fallback MediaPipe model path: {base_path}")
            else:
                 raise RuntimeError("ComfyUI models directory ('models_dir') not found or invalid.")
        except Exception as e:
            logger.error(f"Error determining fallback MediaPipe directory: {e}")
            raise RuntimeError("Could not determine a valid directory for MediaPipe models.") from e

    # Construct the task-specific directory path
    task_dir = os.path.join(base_path, task_type)

    # Ensure the final task directory exists
    try:
        os.makedirs(task_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create MediaPipe task directory {task_dir}: {e}")
        # Propagate error as directory creation is critical
        raise RuntimeError(f"Failed to create required directory: {task_dir}") from e

    return task_dir

def _download_url_to_file(url: str, save_path: str):
    """Downloads a file with progress indication."""
    logger.info(f"Downloading model from {url} to {save_path}...")
    try:
        with urllib.request.urlopen(url) as response, open(save_path, 'wb') as out_file:
            total_size_header = response.getheader('Content-Length')
            total_size = int(total_size_header) if total_size_header else None
            bytes_downloaded = 0
            block_size = 8192
            last_print_bytes = 0
            print_interval = 1 * 1024 * 1024 # Print every 1MB

            if total_size:
                logger.info(f"  File size: {total_size / (1024*1024):.2f} MB")
            else:
                logger.warning("  File size unknown.")

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                bytes_downloaded += len(buffer)

                # Progress reporting (less frequent)
                if bytes_downloaded - last_print_bytes > print_interval or (total_size and bytes_downloaded == total_size):
                    progress_str = f"  Downloaded: {bytes_downloaded / (1024*1024):.2f} MB"
                    if total_size:
                        progress = (bytes_downloaded / total_size) * 100
                        progress_str += f" / {total_size / (1024*1024):.2f} MB ({progress:.1f}%)"
                    # Use print with carriage return for inline update
                    print(progress_str, end='          \r') 
                    last_print_bytes = bytes_downloaded

        print() # Newline after download finishes
        logger.info(f"Download complete: {save_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        logger.error(traceback.format_exc())
        # Clean up potentially corrupted file
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
                logger.info(f"Removed incomplete download: {save_path}")
            except Exception as rm_err:
                logger.error(f"Error removing incomplete download {save_path}: {rm_err}")
        return False

def get_model_path(task_type: str, model_variant: str) -> Optional[str]:
    """Gets the full path to the model file, downloading if necessary."""
    if task_type not in MEDIAPIPE_MODELS:
        logger.error(f"Unknown MediaPipe task type: '{task_type}'")
        return None
    if model_variant not in MEDIAPIPE_MODELS[task_type]:
        logger.error(f"Unknown model variant '{model_variant}' for task '{task_type}'")
        # Log available variants for the task
        available = list(MEDIAPIPE_MODELS[task_type].keys())
        logger.info(f"Available variants for {task_type}: {available}")
        return None

    model_details = MEDIAPIPE_MODELS[task_type][model_variant]
    model_filename = model_details["filename"]
    model_url = model_details["url"]
    
    # Get the specific directory for this task
    try:
        model_dir = get_mediapipe_model_dir(task_type)
    except Exception as e:
         logger.error(f"Cannot determine model directory for {task_type}: {e}")
         return None
         
    model_path = os.path.join(model_dir, model_filename)

    # Check and download
    if not os.path.exists(model_path):
        logger.info(f"Model not found locally at {model_path}. Attempting download.")
        if not _download_url_to_file(model_url, model_path):
            return None # Download failed
    else:
        logger.debug(f"Found model locally: {model_path}")

    return model_path

# --- Public Functions for Node Wrapper --- 

def get_available_tasks() -> List[str]:
    """Get a list of available MediaPipe Vision task types from the definition."""
    return list(MEDIAPIPE_MODELS.keys())

def get_available_models(task_type: str) -> List[str]:
    """Get a list of available model variants for a specific task."""
    if task_type in MEDIAPIPE_MODELS:
        return list(MEDIAPIPE_MODELS[task_type].keys())
    else:
        logger.warning(f"Task type '{task_type}' not found in model definitions.")
        return []

def get_model_description(task_type: str, model_variant: str) -> str:
    """Get the description for a specific model variant.
    
    Args:
        task_type: The MediaPipe task type
        model_variant: The specific model variant
        
    Returns:
        The description string if found, or a default message if not available
    """
    if task_type in MEDIAPIPE_MODELS and model_variant in MEDIAPIPE_MODELS[task_type]:
        return MEDIAPIPE_MODELS[task_type][model_variant].get(
            "description", "No description available"
        )
    return "Model description not available"

# --- Model Loader Class (Simplified) --- 

class MediaPipeModelLoader:
    """Helper class to manage loading state and info.
       Used temporarily by the loader node to get model info.
    """
    
    def __init__(self):
        self.model_path = None
        self.task_type = None
        self.model_variant = None
        self.model_loaded = False
        self.model_info = {}
    
    def load_model(self, task_type: str, model_variant: str) -> bool:
        """Sets model info and triggers path retrieval (incl. download)."""
        logger.debug(f"MediaPipeModelLoader: Loading {task_type}/{model_variant}")
        self.task_type = task_type
        self.model_variant = model_variant
        self.model_path = get_model_path(task_type, model_variant)
        
        if self.model_path and os.path.exists(self.model_path):
            self.model_loaded = True
            self.model_info = {
                'task_type': self.task_type,
                'model_variant': self.model_variant,
                'model_path': self.model_path,
                'description': get_model_description(task_type, model_variant)
            }
            logger.debug(f"Model info set: {self.model_info}")
            return True
        else:
            logger.error(f"Failed to get or validate model path for {task_type}/{model_variant}")
            self.model_loaded = False
            self.model_info = {}
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        # Ensure info is consistent with loaded state
        if self.model_loaded:
             # Add the description to the model info
             if self.task_type and self.model_variant:
                 description = get_model_description(self.task_type, self.model_variant)
                 self.model_info['description'] = description
             return self.model_info
        else:
             # Return minimal info or empty dict if not loaded
             return {
                 'task_type': self.task_type,
                 'model_variant': self.model_variant,
                 'model_path': None # Indicate path is not valid/loaded
             }

# Removed original get_model_path, download_model etc. from global scope
# They are now integrated into the new get_model_path function. 