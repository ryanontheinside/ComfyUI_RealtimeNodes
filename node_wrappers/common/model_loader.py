"""Node wrapper for MediaPipe Vision model loader."""

# Import necessary components from the source directory
from ...src.model_loader import MediaPipeModelLoader, get_available_models
import logging
import traceback # For better error logging

# Add import for MEDIAPIPE_MODELS dictionary and the get_model_description function
from ...src.model_loader import MEDIAPIPE_MODELS, get_model_description

logger = logging.getLogger(__name__)

# Define a base class for MediaPipe model loading nodes
class MediaPipeModelLoaderBaseNode:
    """Base ComfyUI node for loading MediaPipe Vision models.
    
    Inheriting classes MUST define the class variable TASK_TYPE.

    This node handles selecting the model variant for a specific task,
    triggers the download/loading via the src.model_loader,
    and returns a dictionary containing model information needed by other nodes.
    """
    
    # Define common category for all MediaPipe loaders
    CATEGORY = "MediaPipeVision/Loaders"
    # Inheriting classes MUST set this
    TASK_TYPE = None # e.g., "face_detector", "hand_landmarker"

    @classmethod
    def INPUT_TYPES(cls):
        """Define input types for the node."""
        if cls.TASK_TYPE is None:
             raise NotImplementedError("Subclasses of MediaPipeModelLoaderBaseNode must define TASK_TYPE")

        task_variants = []
        default_variant = "default" # Sensible default

        try:
            task_variants = get_available_models(cls.TASK_TYPE)
            if not task_variants:
                 logger.warning(f"No model variants found for task '{cls.TASK_TYPE}'. Defaulting to ['default'].")
                 task_variants = ["default"] # Fallback variants
            default_variant = task_variants[0] if task_variants else default_variant

        except Exception as e:
            logger.error(f"Failed to dynamically get available MediaPipe models for task {cls.TASK_TYPE}: {e}\n{traceback.format_exc()}")
            # Use hardcoded fallbacks if dynamic loading fails
            task_variants = ["default"]
            default_variant = "default"

        # Create tooltip with model descriptions
        tooltip = f"Select model variant for {cls.TASK_TYPE}."
        
        # Add model descriptions if available
        tooltip += "\n\nModel Variants:"
        for variant in task_variants:
            description = get_model_description(cls.TASK_TYPE, variant)
            tooltip += f"\n- {variant}: {description}"

        return {
            "required": {
                # Dropdown for selecting the model variant for the specific task
                "model_variant": (task_variants, {
                    "default": default_variant,
                    "tooltip": tooltip
                }),
            }
        }
    
    # Return a dictionary with model info, not the loader object itself
    RETURN_TYPES = ("MEDIAPIPE_MODEL_INFO",) # Default, can be overridden by subclasses for specificity
    RETURN_NAMES = ("model_info",)
    FUNCTION = "load_and_get_info"

    def load_and_get_info(self, model_variant: str): # Removed task_type argument
        """
        Loads the specified MediaPipe model using the src loader
        and returns a dictionary with model information (path, task, variant).
        Uses the TASK_TYPE defined by the subclass.
        """
        if self.TASK_TYPE is None:
             raise NotImplementedError("TASK_TYPE not defined in the subclass.")

        task_type = self.TASK_TYPE # Use the class variable

        logger.info(f"Attempting to load MediaPipe model: Task='{task_type}', Variant='{model_variant}'")
        
        # Use a temporary loader instance just to trigger the load/download logic
        # The actual detector nodes will instantiate their own detectors using the path
        loader = MediaPipeModelLoader()
        # Pass the class's task_type here
        success = loader.load_model(task_type, model_variant)
        
        if not success:
            error_msg = f"Failed to load MediaPipe model for Task='{task_type}', Variant='{model_variant}'"
            logger.error(error_msg)
            # It's often better to raise an error to stop the workflow
            raise RuntimeError(error_msg)
        
        # Get the essential model information
        model_info = loader.get_model_info() # Should return {'model_path': ..., 'task_type': ..., 'model_variant': ...}
        
        # Ensure the info dict is populated correctly - get_model_info should ideally return all needed keys
        if 'model_path' not in model_info or not model_info['model_path']:
             raise RuntimeError(f"Failed to get valid model path from loader for {task_type}/{model_variant}")
        if 'task_type' not in model_info: model_info['task_type'] = task_type
        if 'model_variant' not in model_info: model_info['model_variant'] = model_variant
             
        logger.info(f"Successfully loaded model info: Path='{model_info.get('model_path')}', Task='{model_info.get('task_type')}', Variant='{model_info.get('model_variant')}'")
        
        # Return the dictionary containing model path and other relevant info
        # The specific RETURN_TYPES are handled by the subclass definition
        return (model_info,)

# Base class should not be exposed directly in mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Removed MediaPipeModelLoaderNode which returned the loader instance.
# Removed CUSTOM_INPUTS_UPDATE as it's not standard ComfyUI.
# Task-specific loaders will inherit from MediaPipeModelLoaderBaseNode. 