from ..base.stream_base import StreamProcessorRegistry
from ..processors.audio_processor import AudioProcessor
import numpy as np

class AudioStreamAdapter:
    """Adapter node that connects ComfyStream audio to our processor"""
    
    CATEGORY = "real-time/control/audio"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "update"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "buffer_size_ms": ("FLOAT", {
                    "default": 500.0,
                    "min": 20.0,  # Minimum one frame
                    "max": 2000.0,  # 2 seconds max
                    "step": 20.0,
                    "tooltip": "Audio buffer size in milliseconds"
                })
            }
        }
    
    def __init__(self):
        self.processor = None
    
    def update(self, audio: np.ndarray, buffer_size_ms: float):
        # Initialize processor if needed or if buffer size changed
        if (self.processor is None or 
            self.processor.buffer_size_samples != int(buffer_size_ms * 48)):
            self.processor = AudioProcessor(
                sample_rate=48000,
                buffer_size_ms=buffer_size_ms
            )
            # Register/update the processor
            StreamProcessorRegistry.register_processor("audio", self.processor)
        
        # Process the audio buffer
        if audio is not None:
            features = self.processor.process_buffer(audio)
            # Update the global feature cache
            StreamProcessorRegistry.update_features("audio", features)
        
        # Pass through the audio buffer
        return (audio,) 