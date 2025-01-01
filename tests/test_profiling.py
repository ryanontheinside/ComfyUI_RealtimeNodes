import cProfile
import pstats
import numpy as np
import cv2
import sys
import os
import json
import time
import torch
from functools import wraps
from pathlib import Path
import line_profiler
import tracemalloc
from io import StringIO

# Add parent directory to path to import our modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import our modules
from detectors.detectors import MotionDetector, BrightnessDetector
from detectors.detector_nodes import MotionDetectorNode, BrightnessDetectorNode, RegionOfInterest
from controls.detector_controls import FloatDetectionControl, IntDetectionControl, StringDetectionControl
from controls.sequence_controls import FloatSequence
from base.detector_base import SharedProcessing

class ProfilingResults:
    def __init__(self, output_dir="profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {
            "timing": {},
            "memory": {},
            "line_profiles": {},
            "summary": {}
        }
    
    def add_timing(self, section, data):
        self.results["timing"][section] = data
    
    def add_memory(self, section, data):
        self.results["memory"][section] = data
    
    def add_line_profile(self, section, stats):
        # Capture line profiler output
        output = StringIO()
        stats.print_stats(output)
        self.results["line_profiles"][section] = output.getvalue()
    
    def add_summary(self, section, data):
        self.results["summary"][section] = data
    
    def save(self):
        # Save JSON-compatible data
        with open(self.output_dir / "profiling_summary.json", "w") as f:
            summary_data = {
                "timing": self.results["timing"],
                "memory": self.results["memory"],
                "summary": self.results["summary"]
            }
            json.dump(summary_data, f, indent=2)
        
        # Save detailed line profiles
        with open(self.output_dir / "line_profiles.txt", "w") as f:
            for section, profile in self.results["line_profiles"].items():
                f.write(f"\n{'='*80}\n")
                f.write(f"Profile for: {section}\n")
                f.write(f"{'='*80}\n")
                f.write(profile)

def create_test_images(size=(512, 512)):
    """Create test images that will exercise different code paths"""
    # Create RGB test images (HWC format)
    images = {
        "uniform": np.ones((*size, 3), dtype=np.uint8) * 128,  # 3-channel RGB
        "random": np.random.randint(0, 255, (*size, 3), dtype=np.uint8),  # 3-channel RGB
        "moving": []
    }
    
    # Create sequence of images with simulated motion
    base = np.zeros((*size, 3), dtype=np.uint8)  # 3-channel RGB
    for i in range(30):
        frame = base.copy()
        x1 = int(size[1] * (i / 30))
        x2 = int(size[1] * ((30-i) / 30))
        # Draw in RGB order since we're not using cv2.imshow
        cv2.rectangle(frame, (x1, 100), (x1 + 50, 150), (255, 255, 255), -1)  # White
        cv2.rectangle(frame, (x2, 300), (x2 + 50, 350), (128, 128, 128), -1)  # Gray
        cv2.circle(frame, (x1, x2), 30, (200, 200, 200), -1)  # Light gray
        images["moving"].append(frame)
    
    # Debug first frame
    print(f"First moving frame shape: {images['moving'][0].shape}")
    print(f"First moving frame dtype: {images['moving'][0].dtype}")
    print(f"First moving frame range: {images['moving'][0].min()}-{images['moving'][0].max()}")
    print(f"First moving frame channels: {images['moving'][0][0,0]}")  # Show first pixel's channels
    
    return images

class CodeProfiler:
    def __init__(self):
        self.line_profiler = line_profiler.LineProfiler()
        self.test_images = create_test_images()
        self.iterations = 1000
        self.results = ProfilingResults()
        
        # Set up line profiling for critical functions
        self.setup_profiling()
    
    def setup_profiling(self):
        """Set up line profiling for critical code paths"""
        # Core detection logic
        self.line_profiler.add_function(MotionDetector.detect)
        self.line_profiler.add_function(BrightnessDetector.detect)
        self.line_profiler.add_function(SharedProcessing.get_shared_data)
        
        # Control node processing
        self.line_profiler.add_function(FloatDetectionControl.process_detection_base)
        self.line_profiler.add_function(FloatDetectionControl._process_action)
        
        # ROI processing
        self.line_profiler.add_function(RegionOfInterest.update)
    
    def profile_detector_nodes(self):
        """Profile detector node operations"""
        print(f"Profiling detector nodes...")
        
        # Create test inputs that mimic ComfyUI node inputs
        image = self.test_images["moving"][0]
        print(f"Original image shape: {image.shape}")
        print(f"Original image dtype: {image.dtype}")
        print(f"Original image range: {image.min()}-{image.max()}")
        print(f"Original image channels: {image[0,0]}")  # Show first pixel's channels
        
        # ComfyUI IMAGE inputs are BHWC format, float32, range [0,1]
        image_tensor = torch.from_numpy(image).float() / 255.0  # Keep HWC format
        print(f"Before batch dim: {image_tensor.shape}")
        print(f"Before batch dtype: {image_tensor.dtype}")
        print(f"Before batch range: {image_tensor.min()}-{image_tensor.max()}")
        print(f"Before batch channels: {image_tensor[0,0]}")
        
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension -> BHWC
        print(f"After batch dim: {image_tensor.shape}")
        print(f"After batch dtype: {image_tensor.dtype}")
        print(f"After batch range: {image_tensor.min()}-{image_tensor.max()}")
        print(f"After batch channels: {image_tensor[0,0,0]}")
        
        # Debug numpy conversion
        test_frame = (image_tensor[0] * 255).cpu().numpy().astype(np.uint8)
        print(f"Test numpy conversion shape: {test_frame.shape}")
        print(f"Test numpy conversion dtype: {test_frame.dtype}")
        print(f"Test numpy conversion range: {test_frame.min()}-{test_frame.max()}")
        print(f"Test numpy conversion channels: {test_frame[0,0]}")  # Show first pixel's channels
        
        # Create mask tensor - ComfyUI expects BHW format
        mask = np.ones((512, 512), dtype=np.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # Add batch dimension -> BHW
        
        # Profile motion detector node
        motion_node = MotionDetectorNode()
        motion_detector = motion_node.update(threshold=0.1, blur_size=5, always_execute=True)[0]
        
        # Profile brightness detector node
        brightness_node = BrightnessDetectorNode()
        brightness_detector = brightness_node.update(threshold=0.5, use_average=True, always_execute=True)[0]
        
        # Create ROI chain
        roi_node = RegionOfInterest()
        roi_inputs = {
            "mask": mask_tensor,
            "detector": motion_detector,
            "action": "add",
            "value": 0.1,
            "always_execute": True
        }
        roi_chain = roi_node.update(**roi_inputs)[0]
        
        # Profile detection controls
        controls = [
            (FloatDetectionControl(), "float_control", {
                "minimum_value": 0.0,
                "maximum_value": 1.0,
                "starting_value": 0.5
            })
        ]
        
        self.line_profiler.enable()
        tracemalloc.start()
        
        timing_data = {}
        for control, name, params in controls:
            start_time = time.perf_counter()
            
            for _ in range(self.iterations):
                inputs = {
                    "image": image_tensor,
                    "roi_chain": roi_chain,
                    "always_execute": True,
                    **params
                }
                control.update(**inputs)
            
            timing_data[name] = time.perf_counter() - start_time
        
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        self.line_profiler.disable()
        
        # Save results
        self.results.add_timing("detector_nodes", timing_data)
        self.results.add_memory("detector_nodes", [str(stat) for stat in snapshot.statistics('lineno')[:10]])
        self.results.add_line_profile("detector_nodes", self.line_profiler)
    
    def profile_core_detection(self):
        """Profile core detection algorithms"""
        print(f"Profiling core detection...")
        
        motion_detector = MotionDetector()
        motion_detector.setup(threshold=0.1, blur_size=5)
        
        brightness_detector = BrightnessDetector()
        brightness_detector.setup(threshold=0.5, use_average=True)
        
        mask = np.ones((512, 512), dtype=np.float32)
        
        self.line_profiler.enable()
        tracemalloc.start()
        
        timing_data = {"motion": 0, "brightness": 0}
        
        # Profile motion detection
        start_time = time.perf_counter()
        for _ in range(self.iterations // len(self.test_images["moving"])):
            state = {}
            for i in range(len(self.test_images["moving"]) - 1):
                current = self.test_images["moving"][i]
                # Convert to RGB for OpenCV
                current = cv2.cvtColor(current, cv2.COLOR_BGR2RGB)
                shared_data = SharedProcessing.get_shared_data(current)
                preprocessed = motion_detector.preprocess(current, shared_data)
                state = {
                    "preprocessed": preprocessed,
                    "shared": shared_data,
                    "y_offset": 0,
                    "x_offset": 0
                }
                motion_detector.detect(current, mask, state)
        timing_data["motion"] = time.perf_counter() - start_time
        
        # Profile brightness detection
        start_time = time.perf_counter()
        for img_type in ["uniform", "random"]:
            image = self.test_images[img_type]
            # Convert to RGB for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for _ in range(self.iterations // 2):
                shared_data = SharedProcessing.get_shared_data(image)
                preprocessed = brightness_detector.preprocess(image, shared_data)
                state = {
                    "preprocessed": preprocessed,
                    "shared": shared_data,
                    "y_offset": 0,
                    "x_offset": 0
                }
                brightness_detector.detect(image, mask, state)
        timing_data["brightness"] = time.perf_counter() - start_time
        
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        self.line_profiler.disable()
        
        # Save results
        self.results.add_timing("core_detection", timing_data)
        self.results.add_memory("core_detection", [str(stat) for stat in snapshot.statistics('lineno')[:10]])
        self.results.add_line_profile("core_detection", self.line_profiler)
    
    def run_profiling(self):
        """Run all profiling tests"""
        start_time = time.time()
        
        self.profile_core_detection()
        self.profile_detector_nodes()
        
        total_time = time.time() - start_time
        self.results.add_summary("execution", {
            "total_time": total_time,
            "iterations": self.iterations,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Save all results
        self.results.save()
        print(f"\nProfiling complete. Results saved to {self.results.output_dir}/")

if __name__ == "__main__":
    profiler = CodeProfiler()
    profiler.run_profiling() 