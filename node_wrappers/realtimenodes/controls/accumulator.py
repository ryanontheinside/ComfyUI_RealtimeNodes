import torch
import numpy as np

from ....src.realtimenodes.control_base import ControlNodeBase
from ....src.utils.general import AlwaysEqualProxy


class BaseAccumulatorNode(ControlNodeBase):
    """
    Base class for accumulator nodes that collect inputs over multiple workflow runs.
    Subclasses should override the process_input and get_output methods to handle specific data types.
    """

    CATEGORY = "Realtime Nodes/control/accumulators"
    FUNCTION = "update"
    DESCRIPTION = "Base class for accumulators that collect inputs over multiple workflow runs"
    
    # These should be overridden by subclasses
    NODE_TYPE_NAME = "Base"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            # Input type will be defined by subclasses
            "batch_size": ("INT", {
                "default": 5,
                "min": 1,
                "max": 1000,
                "step": 1,
                "tooltip": "Number of inputs to accumulate before outputting"
            }),
            "pad_incomplete_batch": ("BOOLEAN", {
                "default": False,
                "tooltip": "If True, duplicate values to reach batch_size when not enough values are accumulated"
            }),
            "reset_after_batch": ("BOOLEAN", {
                "default": False,
                "tooltip": "If True, clears accumulated values after outputting a complete batch"
            }),
            "add_to_batch": ("BOOLEAN", {
                "default": True,
                "tooltip": "If False, the current input will not be added to the batch"
            }),
            "reset_batch": ("BOOLEAN", {
                "default": False,
                "tooltip": "If True, clears all accumulated values"
            })
        })
        return inputs


class AnyAccumulatorNode(BaseAccumulatorNode):
    """
    Node that accumulates inputs of any type over multiple workflow runs.
    """
    
    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("accumulated_values",)
    NODE_TYPE_NAME = "Any"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"]["input_value"] = (AlwaysEqualProxy("*"), {"tooltip": "Any value to accumulate"})
        return inputs
    
    def update(self, input_value, batch_size, pad_incomplete_batch, 
               reset_after_batch, add_to_batch, reset_batch, always_execute=True, unique_id=None):
        """
        For general objects, we still use a list-based approach
        """
        # Use the unique_id directly for state management
        state_key = f"__accumulator_{self.NODE_TYPE_NAME}__"
        
        # Get current state or initialize empty
        state = self.get_state({state_key: []}, unique_id)
        accumulated = state[state_key]
        
        # Handle reset if requested
        if reset_batch:
            accumulated = []
        # Add current input to batch if requested
        elif add_to_batch:
            # When batch is already full, remove oldest item first (FIFO queue)
            if len(accumulated) >= batch_size:
                accumulated.pop(0)  # Remove oldest item
            
            # Add new item
            accumulated.append(input_value)
        
        # Check if batch is complete
        batch_complete = len(accumulated) >= batch_size
        
        # Process accumulated values for output
        if batch_complete:
            # Only take the requested batch size
            output_values = accumulated[:batch_size]
        elif pad_incomplete_batch and accumulated:
            # Pad to reach batch_size by duplicating elements
            output_values = []
            idx = 0
            while len(output_values) < batch_size:
                output_values.append(accumulated[idx % len(accumulated)])
                idx += 1
        else:
            # Just use whatever we have
            output_values = accumulated.copy()
        
        # Reset if batch is complete and reset_after_batch is True
        if batch_complete and reset_after_batch:
            accumulated = []
        
        # Update state
        state[state_key] = accumulated
        self.set_state(state, unique_id)
        
        return (output_values,)


class ImageAccumulatorNode(BaseAccumulatorNode):
    """
    Node that accumulates image inputs (BHWC format) over multiple workflow runs.
    """
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    NODE_TYPE_NAME = "Image"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"]["input_value"] = ("IMAGE", {"tooltip": "Image to accumulate (BHWC format)"})
        return inputs
    
    def update(self, input_value, batch_size, pad_incomplete_batch, 
               reset_after_batch, add_to_batch, reset_batch, always_execute=True, unique_id=None):
        """
        Directly batch tensors instead of storing a list
        """
        # Use the unique_id directly for state management
        state_key = f"__accumulator_{self.NODE_TYPE_NAME}__"
        
        # Get current state or initialize empty tensor
        state = self.get_state({
            state_key: None, 
            f"{state_key}_frames": []
        }, unique_id)
        
        # For tensor types, we keep both the batched tensor and a list of individual frames
        # This allows us to easily implement queue-like behavior
        accumulated_tensor = state[state_key]
        frames = state[f"{state_key}_frames"]
        
        # Add batch dimension if needed (HWC -> BHWC)
        if len(input_value.shape) == 3:
            current_input = input_value.unsqueeze(0)
        else:
            current_input = input_value
            
        # Handle reset if requested
        if reset_batch:
            accumulated_tensor = None
            frames = []
        # Auto-reset on size mismatch
        elif add_to_batch and frames and current_input.shape[1:] != frames[0].shape[1:]:
            print(f"[ImageAccumulatorNode] Size mismatch, resetting batch: {frames[0].shape[1:]} vs {current_input.shape[1:]}")
            accumulated_tensor = None
            frames = []
        
        # Add current input to batch if requested
        if add_to_batch:
            # When batch is already full, remove oldest frame (FIFO queue)
            if len(frames) >= batch_size:
                frames.pop(0)  # Remove oldest frame
            
            # Add new frame(s)
            # Handle multi-batch inputs by adding them one by one
            for i in range(current_input.shape[0]):
                frame = current_input[i:i+1]  # Get single frame with batch dimension
                frames.append(frame)
            
            # Rebuild accumulated tensor from frames
            if frames:
                accumulated_tensor = torch.cat(frames, dim=0)
            else:
                accumulated_tensor = None
        
        # Prepare output
        batch_complete = accumulated_tensor is not None and accumulated_tensor.shape[0] >= batch_size
        
        if batch_complete:
            # Only take the requested batch size
            output_tensor = accumulated_tensor[:batch_size]
        elif pad_incomplete_batch and accumulated_tensor is not None and accumulated_tensor.shape[0] > 0:
            # Pad to reach batch_size by duplicating elements
            tensors_to_cat = [accumulated_tensor]
            remaining = batch_size - accumulated_tensor.shape[0]
            
            while remaining > 0:
                # Add a slice of the tensor that's not bigger than what we need
                slice_size = min(accumulated_tensor.shape[0], remaining)
                tensors_to_cat.append(accumulated_tensor[:slice_size])
                remaining -= slice_size
            
            # Concatenate all tensors
            output_tensor = torch.cat(tensors_to_cat, dim=0)
        else:
            # Return what we have
            output_tensor = accumulated_tensor if accumulated_tensor is not None else torch.zeros((0, 0, 0, 3))
        
        # Reset if batch is complete and reset_after_batch is True
        if batch_complete and reset_after_batch:
            accumulated_tensor = None
            frames = []
        
        # Update state
        state[state_key] = accumulated_tensor
        state[f"{state_key}_frames"] = frames
        self.set_state(state, unique_id)
        
        return (output_tensor,)


class MaskAccumulatorNode(BaseAccumulatorNode):
    """
    Node that accumulates mask inputs over multiple workflow runs.
    """
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask_batch",)
    NODE_TYPE_NAME = "Mask"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"]["input_value"] = ("MASK", {"tooltip": "Mask to accumulate"})
        return inputs
    
    def update(self, input_value, batch_size, pad_incomplete_batch, 
               reset_after_batch, add_to_batch, reset_batch, always_execute=True, unique_id=None):
        """
        Directly batch tensors instead of storing a list
        """
        # Use the unique_id directly for state management
        state_key = f"__accumulator_{self.NODE_TYPE_NAME}__"
        
        # Get current state or initialize empty tensor
        state = self.get_state({
            state_key: None, 
            f"{state_key}_frames": []
        }, unique_id)
        
        # For tensor types, we keep both the batched tensor and a list of individual frames
        accumulated_tensor = state[state_key]
        frames = state[f"{state_key}_frames"]
        
        # Add batch dimension if needed (HW -> BHW)
        if len(input_value.shape) == 2:
            current_input = input_value.unsqueeze(0)
        else:
            current_input = input_value
            
        # Handle reset if requested
        if reset_batch:
            accumulated_tensor = None
            frames = []
        # Auto-reset on size mismatch
        elif add_to_batch and frames and current_input.shape[1:] != frames[0].shape[1:]:
            print(f"[MaskAccumulatorNode] Size mismatch, resetting batch: {frames[0].shape[1:]} vs {current_input.shape[1:]}")
            accumulated_tensor = None
            frames = []
        
        # Add current input to batch if requested
        if add_to_batch:
            # When batch is already full, remove oldest frame (FIFO queue)
            if len(frames) >= batch_size:
                frames.pop(0)  # Remove oldest frame
            
            # Add new frame(s)
            # Handle multi-batch inputs by adding them one by one
            for i in range(current_input.shape[0]):
                frame = current_input[i:i+1]  # Get single frame with batch dimension
                frames.append(frame)
            
            # Rebuild accumulated tensor from frames
            if frames:
                accumulated_tensor = torch.cat(frames, dim=0)
            else:
                accumulated_tensor = None
        
        # Prepare output
        batch_complete = accumulated_tensor is not None and accumulated_tensor.shape[0] >= batch_size
        
        if batch_complete:
            # Only take the requested batch size
            output_tensor = accumulated_tensor[:batch_size]
        elif pad_incomplete_batch and accumulated_tensor is not None and accumulated_tensor.shape[0] > 0:
            # Pad to reach batch_size by duplicating elements
            tensors_to_cat = [accumulated_tensor]
            remaining = batch_size - accumulated_tensor.shape[0]
            
            while remaining > 0:
                # Add a slice of the tensor that's not bigger than what we need
                slice_size = min(accumulated_tensor.shape[0], remaining)
                tensors_to_cat.append(accumulated_tensor[:slice_size])
                remaining -= slice_size
            
            # Concatenate all tensors
            output_tensor = torch.cat(tensors_to_cat, dim=0)
        else:
            # Return what we have
            output_tensor = accumulated_tensor if accumulated_tensor is not None else torch.zeros((0, 0))
        
        # Reset if batch is complete and reset_after_batch is True
        if batch_complete and reset_after_batch:
            accumulated_tensor = None
            frames = []
        
        # Update state
        state[state_key] = accumulated_tensor
        state[f"{state_key}_frames"] = frames
        self.set_state(state, unique_id)
        
        return (output_tensor,)


class LatentAccumulatorNode(BaseAccumulatorNode):
    """
    Node that accumulates latent inputs over multiple workflow runs.
    """
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_batch",)
    NODE_TYPE_NAME = "Latent"
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"]["input_value"] = ("LATENT", {"tooltip": "Latent to accumulate"})
        return inputs
    
    def update(self, input_value, batch_size, pad_incomplete_batch, 
               reset_after_batch, add_to_batch, reset_batch, always_execute=True, unique_id=None):
        """
        Directly batch latent tensors instead of storing a list
        """
        # Use the unique_id directly for state management
        state_key = f"__accumulator_{self.NODE_TYPE_NAME}__"
        
        # Get current state or initialize empty
        state = self.get_state({
            f"{state_key}_samples": None,
            f"{state_key}_frames": [],
            f"{state_key}_metadata": {}
        }, unique_id)
        
        accumulated_samples = state[f"{state_key}_samples"]
        frames = state[f"{state_key}_frames"]
        metadata = state[f"{state_key}_metadata"]
        
        # Get samples tensor from input latent
        current_samples = input_value['samples']
        
        # Collect metadata from the first latent
        if not metadata and add_to_batch:
            for key, value in input_value.items():
                if key != 'samples':
                    metadata[key] = value
        
        # Handle reset if requested
        if reset_batch:
            accumulated_samples = None
            frames = []
        # Auto-reset on size mismatch
        elif add_to_batch and frames and current_samples.shape[1:] != frames[0].shape[1:]:
            print(f"[LatentAccumulatorNode] Size mismatch, resetting batch: {frames[0].shape[1:]} vs {current_samples.shape[1:]}")
            accumulated_samples = None
            frames = []
        
        # Add current input to batch if requested
        if add_to_batch:
            # When batch is already full, remove oldest frame (FIFO queue)
            if len(frames) >= batch_size:
                frames.pop(0)  # Remove oldest frame
            
            # Add new frame(s)
            # Handle multi-batch inputs by adding them one by one
            for i in range(current_samples.shape[0]):
                frame = current_samples[i:i+1]  # Get single frame with batch dimension
                frames.append(frame)
            
            # Rebuild accumulated tensor from frames
            if frames:
                accumulated_samples = torch.cat(frames, dim=0)
            else:
                accumulated_samples = None
        
        # Prepare output
        batch_complete = accumulated_samples is not None and accumulated_samples.shape[0] >= batch_size
        
        if batch_complete:
            # Only take the requested batch size
            output_samples = accumulated_samples[:batch_size]
        elif pad_incomplete_batch and accumulated_samples is not None and accumulated_samples.shape[0] > 0:
            # Pad to reach batch_size by duplicating elements
            tensors_to_cat = [accumulated_samples]
            remaining = batch_size - accumulated_samples.shape[0]
            
            while remaining > 0:
                # Add a slice of the tensor that's not bigger than what we need
                slice_size = min(accumulated_samples.shape[0], remaining)
                tensors_to_cat.append(accumulated_samples[:slice_size])
                remaining -= slice_size
            
            # Concatenate all tensors
            output_samples = torch.cat(tensors_to_cat, dim=0)
        else:
            # Return what we have
            output_samples = accumulated_samples if accumulated_samples is not None else torch.zeros((0, 0, 0, 0))
        
        # Reset if batch is complete and reset_after_batch is True
        if batch_complete and reset_after_batch:
            accumulated_samples = None
            frames = []
        
        # Update state
        state[f"{state_key}_samples"] = accumulated_samples
        state[f"{state_key}_frames"] = frames
        state[f"{state_key}_metadata"] = metadata
        self.set_state(state, unique_id)
        
        # Return latent dict with samples
        return ({"samples": output_samples, **metadata},)


