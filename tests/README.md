# ComfyUI RealTimeNodes Profiling Suite

This directory contains comprehensive profiling tests for the ComfyUI RealTimeNodes.

## Features
- CPU and memory profiling for all detector nodes
- Tests with various image sizes (256x256, 512x512, 1024x1024)
- Different ROI configurations
- Sequence processing tests
- Multiple image types (static, noisy, gradient)
- JSON output of timing results
- Detailed memory usage tracking

## Setup
```bash
pip install -r requirements.txt
```

## Running Profiling Tests
```bash
python test_profiling.py
```

This will:
1. Create a test dataset in `test_data/`
2. Run comprehensive tests on all detectors
3. Generate two output files:
   - `detector_profile.stats`: Detailed CPU profiling data
   - `profiling_results.json`: Structured results including timing and memory usage

## Visualizing Results

### CPU Profile Visualization
```bash
snakeviz detector_profile.stats
```

This will open an interactive visualization showing:
- Call graphs
- Time spent in each function
- Number of calls
- Cumulative time

### Memory Profile Analysis
Memory usage data is printed during test execution and saved in the JSON results file.

## Test Coverage
The suite tests:
1. Individual detector nodes
   - Motion detector
   - Brightness detector
2. ROI configurations
   - Full frame
   - Partial frame
   - Different positions
3. Sequence processing
   - Multiple frame sequences
   - Different detector types
4. Image variations
   - Different sizes
   - Different content types
   - Different complexity levels

## Interpreting Results
The `profiling_results.json` file contains:
- Timing data for each test configuration
- Memory usage statistics
- Node execution statistics

Use this data to:
- Identify performance bottlenecks
- Optimize memory usage
- Improve real-time processing capabilities 