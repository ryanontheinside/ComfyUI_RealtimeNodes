from ..base.stream_base import StreamControl
from ..processors.audio_processor import AudioProcessor
import numpy as np

class AudioFeatureControl(StreamControl):
    """Base class for audio-reactive controls"""
    
    CATEGORY = "real-time/control/audio"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float(time.time())  # Always update to get latest audio features

class VolumeControl(AudioFeatureControl):
    """Control node that reacts to audio volume (RMS)"""
    
    DESCRIPTION = "Outputs a value based on audio volume"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "update"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "min_value": ("FLOAT", {
                    "default": 0.0,
                    "min": -10000.0,
                    "max": 10000.0,
                    "step": 0.1
                }),
                "max_value": ("FLOAT", {
                    "default": 1.0,
                    "min": -10000.0,
                    "max": 10000.0,
                    "step": 0.1
                })
            }
        }
    
    def update(self, processor: str, feature: str, smoothing: float,
              min_value: float, max_value: float, always_execute: bool = True):
        # Get current RMS value
        rms = self.get_feature_value(processor, "rms", 0.0)
        
        # Get previous state
        state = self.get_state({"value": min_value})
        current = state["value"]
        
        # Apply smoothing
        smoothed = self.smooth_value(current, rms, smoothing)
        
        # Map to output range
        output = np.interp(smoothed, [0.0, 1.0], [min_value, max_value])
        
        # Update state
        self.set_state({"value": output})
        
        return (float(output),)

class FrequencyBandControl(AudioFeatureControl):
    """Control node that reacts to specific frequency bands"""
    
    DESCRIPTION = "Outputs a value based on energy in specific frequency bands"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "update"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "band_start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 31,
                    "step": 1,
                    "tooltip": "Start index of frequency band (0-31)"
                }),
                "band_end": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 31,
                    "step": 1,
                    "tooltip": "End index of frequency band (0-31)"
                }),
                "min_value": ("FLOAT", {
                    "default": 0.0,
                    "min": -10000.0,
                    "max": 10000.0,
                    "step": 0.1
                }),
                "max_value": ("FLOAT", {
                    "default": 1.0,
                    "min": -10000.0,
                    "max": 10000.0,
                    "step": 0.1
                })
            }
        }
    
    def update(self, processor: str, feature: str, smoothing: float,
              band_start: int, band_end: int, min_value: float, max_value: float,
              always_execute: bool = True):
        # Get spectrum
        spectrum = self.get_feature_value(processor, "spectrum", 
                                        np.zeros(32))
        
        # Calculate average energy in band range
        band_energy = np.mean(spectrum[band_start:band_end+1])
        
        # Get previous state
        state = self.get_state({"value": min_value})
        current = state["value"]
        
        # Apply smoothing
        smoothed = self.smooth_value(current, band_energy, smoothing)
        
        # Map to output range
        output = np.interp(smoothed, [0.0, 1.0], [min_value, max_value])
        
        # Update state
        self.set_state({"value": output})
        
        return (float(output),)

class BeatControl(AudioFeatureControl):
    """Control node that detects beats in audio"""
    
    DESCRIPTION = "Outputs a trigger value on detected beats"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "update"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Onset detection threshold"
                }),
                "trigger_value": ("FLOAT", {
                    "default": 1.0,
                    "min": -10000.0,
                    "max": 10000.0,
                    "step": 0.1,
                    "tooltip": "Value to output when beat detected"
                }),
                "fallback_value": ("FLOAT", {
                    "default": 0.0,
                    "min": -10000.0,
                    "max": 10000.0,
                    "step": 0.1,
                    "tooltip": "Value to output when no beat"
                })
            }
        }
    
    def update(self, processor: str, feature: str, smoothing: float,
              threshold: float, trigger_value: float, fallback_value: float,
              always_execute: bool = True):
        # Get onset strength
        onset = self.get_feature_value(processor, "onset", 0.0)
        
        # Get previous state
        state = self.get_state({
            "value": fallback_value,
            "last_trigger_time": 0.0
        })
        
        current_time = time.time()
        output = fallback_value
        
        # Check if we should trigger
        if onset > threshold:
            # Ensure minimum time between triggers (100ms)
            if current_time - state["last_trigger_time"] > 0.1:
                output = trigger_value
                state["last_trigger_time"] = current_time
        
        # Apply smoothing
        smoothed = self.smooth_value(state["value"], output, smoothing)
        state["value"] = smoothed
        
        # Update state
        self.set_state(state)
        
        return (float(smoothed),)

# Register the audio processor
StreamProcessorRegistry.register_processor(
    "audio",
    AudioProcessor(sample_rate=48000)
) 