import torch
import os
import numpy as np
import sys
import traceback
import cv2
import datetime

from ....src.realtimenodes.control_base import ControlNodeBase

#NOTE: this is a rudimentary spike, and will be brittle

#TODO: use base class state management

class BatchInfo:
    """Custom class to hold batch processing information"""
    def __init__(self, current_index, total_frames, is_last_frame):
        self.current_index = current_index
        self.total_frames = total_frames
        self.is_last_frame = is_last_frame


class BatchImageProcessor(ControlNodeBase):
    """Node that feeds images one by one from a batch into the workflow"""
    
    # Static variables to persist state across executions
    _global_state = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "images": ("IMAGE",),
            "reset": ("BOOLEAN", {"default": False, "tooltip": "Reset the batch processing"}),
        })
        return inputs
    
    RETURN_TYPES = ("IMAGE", "BATCH_INFO")
    RETURN_NAMES = ("image", "batch_info")
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/media/batch"
    DESCRIPTION = "Feeds images one by one from a batch into the workflow. Must be used in conjunction with BatchResultCollector node."
    def update(self, images, reset, always_execute=True, unique_id=None):
        
        # Use class static variable to persist state
        if unique_id not in self.__class__._global_state:
            self.__class__._global_state[unique_id] = {"current_index": 0, "total_frames": 0}
        
        state = self.__class__._global_state[unique_id]
        
        # Get batch size (first dimension of tensor)
        batch_size = images.size(0)
        
        # Reset if requested or if new batch size
        if reset or state["total_frames"] != batch_size:
            print(f"[BatchImageProcessor] Resetting state. Was at index {state['current_index']}/{state['total_frames']}, new total frames: {batch_size}")
            state["current_index"] = 0
            state["total_frames"] = batch_size
        
        # Get current image (preserving batch dimension)
        current_index = state["current_index"]
        current_image = images[current_index:current_index+1]
        
        # Prepare batch info
        is_last_frame = current_index == state["total_frames"] - 1
        batch_info = BatchInfo(
            current_index=current_index,
            total_frames=state["total_frames"],
            is_last_frame=is_last_frame
        )
        
        # Update index for next run
        state["current_index"] = (current_index + 1) % state["total_frames"]
        print(f"[BatchImageProcessor] Processing frame {current_index+1}/{state['total_frames']} (is_last_frame: {is_last_frame})")
        
        return (current_image, batch_info)


class BatchResultCollector(ControlNodeBase):
    """Node that collects batch results and saves to video when complete"""
    
    # Static variables to persist state across executions
    _global_state = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "image": ("IMAGE",),
            "batch_info": ("BATCH_INFO",),
            "filename_prefix": ("STRING", {"default": "batch_video"}),
            "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0}),
            "format": (["mp4", "webm", "gif"], {"default": "mp4"}),
            "reset": ("BOOLEAN", {"default": False}),
        })
        return inputs
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/media/batch"
    OUTPUT_NODE = True
    DESCRIPTION = "Collects frames from a batch and saves them as a video when the last frame is received. If the last frame is not received, the node will block execution until it is received. Must be used in conjunction with BatchImageProcessor node."
    def update(self, image, batch_info, filename_prefix, fps, format, reset, always_execute=True, unique_id=None):
      
        # Create default state structure
        default_state = {
            "collected_frames": [],
            "current_index": -1,
            "total_frames": 0
        }
        
        # Use class static variable to persist state
        if unique_id not in self.__class__._global_state:
            self.__class__._global_state[unique_id] = default_state.copy()
        
        state = self.__class__._global_state[unique_id]
        
        # Ensure all required keys exist in state
        for key in default_state:
            if key not in state:
                state[key] = default_state[key]
        
        # Reset if requested
        if reset:
            print(f"[BatchResultCollector] Resetting state due to reset flag")
            state.clear()
            state.update(default_state.copy())
        
        current_index = batch_info.current_index
        total_frames = batch_info.total_frames
        
        # Update state if new batch
        if state["total_frames"] != total_frames:
            print(f"[BatchResultCollector] New batch detected. Updating total_frames from {state['total_frames']} to {total_frames}")
            state["total_frames"] = total_frames
            state["collected_frames"] = [None] * total_frames
        
        # Store the frame
        if state["current_index"] != current_index:
            state["current_index"] = current_index
            # Ensure the array is large enough
            if len(state["collected_frames"]) <= current_index:
                state["collected_frames"].extend([None] * (current_index - len(state["collected_frames"]) + 1))
            # Make a copy and store the frame
            state["collected_frames"][current_index] = image.clone().cpu()
        
        # Check if all frames collected
        collected_count = sum(1 for frame in state["collected_frames"] if frame is not None)
        all_frames_collected = all(frame is not None for frame in state["collected_frames"])
        print(f"[BatchResultCollector] Collected {collected_count}/{total_frames} frames. All frames collected: {all_frames_collected}")
        
        # If last frame and all collected, save video
        if batch_info.is_last_frame and all_frames_collected:
            print(f"[BatchResultCollector] Last frame received and all frames collected. Saving video...")
            success = self._save_video(state["collected_frames"], filename_prefix, fps, format)
            
            # Reset state after save
            state.clear()
            state.update(default_state.copy())
            
            if success:
                # Stop auto-queue with a deliberate exception to notify the user
                print(f"[BatchResultCollector] Successfully saved video. Stopping auto-queue.")
                raise Exception("Processing complete. Disable auto-queue.")
            
        return (image,)
    
    def _save_frames_as_images(self, frames, filename_prefix, output_dir):
        """Save each frame as an individual image for debugging"""
        try:
            import folder_paths
            from PIL import Image
            
            # Print details of the first frame for debugging
            print(f"[BatchResultCollector] SaveFrames: First frame type: {type(frames[0])}")
            print(f"[BatchResultCollector] SaveFrames: First frame tensor shape: {frames[0].shape}")
            
            # Get video dimensions from the first frame
            first_frame = frames[0]
            if len(first_frame.shape) == 4:  # Handle [B,H,W,C] format
                height, width = first_frame.shape[1], first_frame.shape[2]
            else:  # Handle [H,W,C] format
                height, width = first_frame.shape[0], first_frame.shape[1]
                
            print(f"[BatchResultCollector] SaveFrames dimensions determined: {width}x{height}")
            
            # Generate timestamp for unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get save path
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                f"{filename_prefix}_{timestamp}", output_dir, width, height
            )
            
            print(f"[BatchResultCollector] Saving {len(frames)} individual frames to {full_output_folder}")
            
            for i, frame_tensor in enumerate(frames):
                print(f"[BatchResultCollector] Saving individual frame {i+1}/{len(frames)}")
                
                # Remove batch dimension if present (B,H,W,C) -> (H,W,C)
                if len(frame_tensor.shape) == 4:
                    frame_tensor = frame_tensor.squeeze(0)
                
                # Convert tensor to numpy
                frame_np = torch.clamp(frame_tensor * 255, min=0, max=255).to(torch.uint8).cpu().numpy()
                
                # Create PIL image and save
                img = Image.fromarray(frame_np)
                img_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_{i:04}.png")
                img.save(img_path)
                
            print(f"[BatchResultCollector] Successfully saved {len(frames)} individual frames")
            return True
        except Exception as e:
            print(f"[BatchResultCollector] Error saving individual frames: {str(e)}")
            traceback.print_exc()
            return False
    
    def _save_as_gif(self, frames, filename_prefix, fps, output_dir):
        """Save frames as animated GIF using PIL"""
        try:
            import folder_paths
            from PIL import Image
            
            # Print details of the first frame for debugging
            print(f"[BatchResultCollector] GIF: First frame type: {type(frames[0])}")
            print(f"[BatchResultCollector] GIF: First frame tensor shape: {frames[0].shape}")
            
            # Get video dimensions from the first frame
            first_frame = frames[0]
            if len(first_frame.shape) == 4:  # Handle [B,H,W,C] format
                height, width = first_frame.shape[1], first_frame.shape[2]
            else:  # Handle [H,W,C] format
                height, width = first_frame.shape[0], first_frame.shape[1]
                
            print(f"[BatchResultCollector] GIF dimensions determined: {width}x{height}")
            
            # Generate timestamp for unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get save path
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                f"{filename_prefix}_{timestamp}", output_dir, width, height
            )
            
            file = f"{filename}_{counter:05}.gif"
            file_path = os.path.join(full_output_folder, file)
            
            print(f"[BatchResultCollector] Saving GIF to {file_path}")
            
            # Convert tensors to PIL images
            pil_frames = []
            for i, frame_tensor in enumerate(frames):
                print(f"[BatchResultCollector] Processing GIF frame {i+1}/{len(frames)}")
                
                # Remove batch dimension if present (B,H,W,C) -> (H,W,C)
                if len(frame_tensor.shape) == 4:
                    frame_tensor = frame_tensor.squeeze(0)
                
                # Convert tensor to numpy
                frame_np = torch.clamp(frame_tensor * 255, min=0, max=255).to(torch.uint8).cpu().numpy()
                pil_frames.append(Image.fromarray(frame_np))
            
            # Save as GIF
            duration = int(1000 / fps)  # Duration per frame in milliseconds
            pil_frames[0].save(
                file_path,
                save_all=True,
                append_images=pil_frames[1:],
                optimize=False,
                duration=duration,
                loop=0
            )
            
            print(f"[BatchResultCollector] Successfully saved GIF with {len(frames)} frames to {file_path}")
            return True
        except Exception as e:
            print(f"[BatchResultCollector] Error saving GIF: {str(e)}")
            traceback.print_exc()
            return False
    
    def _save_video(self, frames, filename_prefix, fps, format):
        """Save collected frames as video file using OpenCV"""
        import folder_paths
        
        # Get output directory
        output_dir = folder_paths.get_output_directory()
        
        if format == "gif":
            return self._save_as_gif(frames, filename_prefix, fps, output_dir)
            
        try:
            # Generate timestamp for unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get video dimensions from the first frame
            first_frame = frames[0]
            if len(first_frame.shape) == 4:  # Handle [B,H,W,C] format
                height, width = first_frame.shape[1], first_frame.shape[2]
            else:  # Handle [H,W,C] format
                height, width = first_frame.shape[0], first_frame.shape[1]
            
            # Get save path
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                f"{filename_prefix}_{timestamp}", output_dir, width, height
            )
            
            file = f"{filename}_{counter:05}.{format}"
            file_path = os.path.join(full_output_folder, file)
            
            print(f"[BatchResultCollector] Saving video to {file_path}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        
            # Try with avc1 codec first (H.264)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
            
            # Fall back to mp4v if avc1 fails
            if not out.isOpened():
                print(f"[BatchResultCollector] Warning: Failed to create video with avc1 codec, trying mp4v instead")
                out.release()
                
                if format == 'mp4':
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                else:
                    fourcc = cv2.VideoWriter_fourcc(*'vp90')
                    
                out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    raise Exception(f"Failed to open video writer with any codec for format {format}")
            
            # Write frames
            frames_encoded = 0
            for i, frame_tensor in enumerate(frames):
                # Remove batch dimension if present (B,H,W,C) -> (H,W,C)
                if len(frame_tensor.shape) == 4:
                    frame_tensor = frame_tensor.squeeze(0)
                
                # Convert tensor to numpy array with values in [0-255]
                frame_np = torch.clamp(frame_tensor * 255, min=0, max=255).to(torch.uint8).cpu().numpy()
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(frame_bgr)
                frames_encoded += 1
                
                if frames_encoded % 10 == 0:
                    print(f"[BatchResultCollector] Encoded {frames_encoded}/{len(frames)} frames")
            
            # Release video writer
            out.release()
            
            print(f"[BatchResultCollector] Video saved successfully with {frames_encoded} frames to {file_path}")
            return True
            
        except Exception as e:
            print(f"[BatchResultCollector] Error saving video: {str(e)}")
            traceback.print_exc()
            print("[BatchResultCollector] Falling back to saving individual frames")
            return self._save_frames_as_images(frames, filename_prefix, output_dir) 