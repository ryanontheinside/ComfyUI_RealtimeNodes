import numpy as np
from scipy import signal
from typing import Dict, Any
from ..base.stream_base import StreamDataProcessor, FeatureSpec

class AudioProcessor(StreamDataProcessor):
    """Processor for real-time audio stream data"""
    
    def __init__(self, sample_rate: int = 48000, 
                 buffer_size_ms: float = 500.0,
                 n_bands: int = 32, 
                 min_freq: float = 20.0,
                 max_freq: float = 20000.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.buffer_size_samples = int(buffer_size_ms * (sample_rate / 1000))
        self.n_bands = n_bands
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        # Pre-compute frequency bands
        self.freq_bands = self._get_mel_bands()
        
        # Initialize FFT config - ensure it's not larger than buffer
        self.n_fft = min(2048, self.buffer_size_samples)  
        self.hop_length = self.n_fft // 4
        
        # Window function for FFT
        self.window = signal.windows.hann(self.n_fft)
        
        # Smoothing state
        self.prev_spectrum = np.zeros(self.n_bands)
        self.prev_rms = 0.0
        
    def get_feature_specs(self) -> Dict[str, FeatureSpec]:
        """Define available audio features"""
        return {
            "rms": FeatureSpec(
                type="FLOAT",
                range=(0.0, 1.0),
                description="Root mean square amplitude"
            ),
            "spectrum": FeatureSpec(
                type="TENSOR",
                shape=(self.n_bands,),
                range=(0.0, 1.0),
                description="Mel-scaled frequency spectrum"
            ),
            "onset": FeatureSpec(
                type="FLOAT",
                range=(0.0, 1.0),
                description="Onset detection strength"
            ),
            "pitch": FeatureSpec(
                type="FLOAT",
                range=(0.0, 1.0),
                description="Estimated dominant pitch"
            )
        }
    
    def _get_mel_bands(self) -> np.ndarray:
        """Create mel-scaled frequency bands"""
        return librosa.mel_frequencies(
            n_mels=self.n_bands,
            fmin=self.min_freq,
            fmax=self.max_freq,
            htk=True
        )
    
    def _compute_spectrum(self, buffer: np.ndarray) -> np.ndarray:
        """Compute mel-scaled spectrum"""
        # Pad buffer if needed
        if len(buffer) < self.n_fft:
            buffer = np.pad(buffer, (0, self.n_fft - len(buffer)))
        
        # Compute STFT
        stft = np.abs(librosa.stft(
            buffer,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=False
        ))
        
        # Convert to mel scale
        mel = librosa.feature.melspectrogram(
            S=stft,
            sr=self.sample_rate,
            n_mels=self.n_bands,
            fmin=self.min_freq,
            fmax=self.max_freq
        )
        
        # Convert to dB scale and normalize
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
        
        # Average across time
        return np.mean(mel_norm, axis=1)
    
    def _compute_onset(self, buffer: np.ndarray) -> float:
        """Detect onset strength"""
        onset_env = librosa.onset.onset_strength(
            y=buffer,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        return float(np.mean(onset_env))
    
    def _compute_pitch(self, buffer: np.ndarray) -> float:
        """Estimate dominant pitch"""
        pitches, magnitudes = librosa.piptrack(
            y=buffer,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.min_freq,
            fmax=self.max_freq
        )
        
        # Get pitch with highest magnitude
        pitch_idx = magnitudes.argmax()
        return float(pitches.flatten()[pitch_idx])
    
    def process_buffer(self, buffer: np.ndarray) -> Dict[str, Any]:
        """Process audio buffer and compute features"""
        # Validate buffer
        if buffer is None:
            return {
                "rms": 0.0,
                "spectrum": np.zeros(self.n_bands),
                "onset": 0.0,
                "pitch": 0.0
            }
            
        # Ensure buffer is the right type and normalized
        if buffer.dtype != np.int16:
            raise ValueError(f"Expected int16 buffer, got {buffer.dtype}")
        buffer = buffer.astype(np.float32) / np.iinfo(np.int16).max
        
        # Ensure buffer size
        if len(buffer) != self.buffer_size_samples:
            if len(buffer) < self.buffer_size_samples:
                buffer = np.pad(buffer, (0, self.buffer_size_samples - len(buffer)))
            else:
                buffer = buffer[:self.buffer_size_samples]
        
        # Compute RMS
        rms = np.sqrt(np.mean(buffer**2))
        
        # Compute spectrum
        spectrum = self._compute_spectrum(buffer)
        
        # Apply temporal smoothing
        spectrum = 0.5 * spectrum + 0.5 * self.prev_spectrum
        rms = 0.5 * rms + 0.5 * self.prev_rms
        
        # Update previous values
        self.prev_spectrum = spectrum
        self.prev_rms = rms
        
        # Compute additional features
        onset = self._compute_onset(buffer)
        pitch = self._compute_pitch(buffer)
        
        return {
            "rms": float(rms),
            "spectrum": spectrum,
            "onset": float(onset),
            "pitch": float(pitch)
        } 