"""Utilities for downloading MediaPipe models."""

import os
import requests
import logging
from tqdm import tqdm
from typing import Dict, Optional, Any, Tuple

from .model_utils import MEDIAPIPE_MODELS, get_model_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_model(task_type: str, model_variant: str) -> Optional[str]:
    """Download a MediaPipe model for a specific task and variant.
    
    Args:
        task_type: The task category (e.g., 'face_mesh', 'pose_detection')
        model_variant: The specific model variant (e.g., 'standard', 'lite')
        
    Returns:
        Path to the downloaded model file, or None if download failed
    """
    # Check if the task and variant exist
    if task_type not in MEDIAPIPE_MODELS:
        logger.error(f"Task '{task_type}' not found in available models.")
        return None
    
    if model_variant not in MEDIAPIPE_MODELS[task_type]:
        logger.error(f"Model variant '{model_variant}' not found for task '{task_type}'.")
        return None
    
    # Get model details
    model_info = MEDIAPIPE_MODELS[task_type][model_variant]
    url = model_info["url"]
    filename = model_info["filename"]
    
    # Get the full path for the model
    output_path = get_model_path(task_type, filename)
    
    # If model already exists, return the path
    if os.path.exists(output_path):
        logger.info(f"Using existing model: {output_path}")
        return output_path
    
    # Download the model
    logger.info(f"Downloading {task_type}/{model_variant} model from {url}...")
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Stream the download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get content length for progress bar if available
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8 KB
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"{task_type}/{model_variant}") as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Successfully downloaded {task_type}/{model_variant} to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error downloading {task_type}/{model_variant}: {str(e)}")
        
        # Clean up partial download if it exists
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        
        return None


def get_model_options(task_type: Optional[str] = None) -> Dict[str, Any]:
    """Get available model options for a specific task or all tasks.
    
    Args:
        task_type: The task category to get options for, or None for all tasks
        
    Returns:
        Dictionary of available models and variants
    """
    if task_type is None:
        # Return all tasks and their variants
        return {task: list(variants.keys()) for task, variants in MEDIAPIPE_MODELS.items()}
    
    # Return variants for specific task
    if task_type in MEDIAPIPE_MODELS:
        return {task_type: list(MEDIAPIPE_MODELS[task_type].keys())}
    
    # Task not found
    return {}


def get_all_task_types() -> list:
    """Get a list of all available task types.
    
    Returns:
        List of task type strings
    """
    return list(MEDIAPIPE_MODELS.keys())


def get_model_variants(task_type: str) -> list:
    """Get a list of all available model variants for a specific task.
    
    Args:
        task_type: The task category
        
    Returns:
        List of model variant strings, or empty list if task not found
    """
    if task_type in MEDIAPIPE_MODELS:
        return list(MEDIAPIPE_MODELS[task_type].keys())
    return []


def get_model_info(task_type: str, model_variant: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Get information about a specific model.
    
    Args:
        task_type: The task category
        model_variant: The specific model variant
        
    Returns:
        Tuple containing:
            - Path to the model file (downloaded if needed)
            - Dictionary with model metadata (or None if task/variant not found)
    """
    # Check if the task and variant exist
    if task_type not in MEDIAPIPE_MODELS or model_variant not in MEDIAPIPE_MODELS[task_type]:
        return None, None
    
    # Get model details
    model_info = MEDIAPIPE_MODELS[task_type][model_variant].copy()
    
    # Download the model if needed and get the path
    model_path = download_model(task_type, model_variant)
    if not model_path:
        return None, None
    
    # Add the model path to the info
    model_info["model_path"] = model_path
    
    return model_path, model_info 