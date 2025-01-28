# ComfyUI Real-Time Nodes ⏱️

A growing suite of nodes for real-time ComfyUI workflows. Features include value animation, motion detection and tracking, sequence control, and more. These nodes update their outputs on each workflow execution, making them perfect for real-time applications like [ComfyStream](https://github.com/yondonfu/comfystream) that execute the workflow once per frame. 

The intention for this repository is to build a suite of nodes that can be used in the burgeoning real-time diffusion space. Contributions are welcome!

## Nodes

### Value Controls 🎚️
- **FloatControl**: Outputs a floating point value that changes over time using various patterns (sine wave, bounce, random walk, etc).
- **IntControl**: Same as FloatControl but outputs integer values.
- **StringControl**: Cycles through a list of strings using the same movement patterns.

### Sequence Controls 📽️
- **FloatSequence**: Cycles through a comma-separated list of float values.
- **IntSequence**: Cycles through a comma-separated list of integer values.
- **StringSequence**: Cycles through a list of strings (one per line).

### Motion Controls 🎮
- **MotionController**: Advanced float-based motion control for smooth animations.
- **IntegerMotionController**: Integer-based motion control for discrete value animations.
- **ROINode**: Region of Interest node for motion tracking and control.

### Utility Controls 🛠️
- **FPSMonitor**: Generates an FPS overlay as an image and mask, useful for monitoring performance.
- **QuickShapeMask**: Rapidly generate shape masks (circle, square) with customizable dimensions.
- **DTypeConverter**: Convert masks between different data types (float16, uint8, float32, float64).
- **FastWebcamCapture**: High-performance webcam capture node with resizing capabilities.
- **SimilarityFilter**: Filter images based on the similarity of previous frames and block execution if the similarity is too high.
### VAE Tools 🖼️
- **TAESDVaeEncode**: TAESD VAE encoding node.
- **TAESDVaeDecode**: TAESD VAE decoding node.

## Movement Patterns 🔄

All value and motion controls support various movement patterns:
- **Sine**: Smooth sinusoidal motion
- **Triangle**: Linear interpolation with smooth direction changes
- **Sawtooth**: Linear interpolation with sharp resets
- **Square**: Instant transitions between min/max values
- **Static**: No movement (constant value)
- **and more**

## Usage 📖

### Value Controls
Connect any value control node to the input of the node you want to animate. These nodes use movement patterns like sine, bounce, etc. to smoothly transition between values.

### Motion Controls
Use motion controllers for more advanced animation control:
- Set minimum/maximum values
- Control steps per cycle
- Choose movement patterns
- Apply to any numeric parameter

### Sequence Controls
Sequence controls allow you to specify exact values to cycle through. You can control:
- Steps per item: How many frames to show each value
- Sequence mode: forward, reverse, pingpong, or random

### FPS Monitor
Outputs an image and mask showing current and average FPS. Useful for performance monitoring in real-time workflows.

## Examples 🎬

### Value Control Demo
In this demo we are controlling the width and height of a shape mask with an Int Control node. Imagine controlling the denoise on a KSampler with a Float Control, though! Or CFG in StreamDiffusion!
![Value Control Demo](assets/ValueControl.gif)

### Motion Detection Blur Demo
Example of motion-based active blur effect running in ComfyStream. The action here is to control the blur amount of the frame here, but imagine if the action was "fire weapon":
![Motion Active Blur Demo](https://github.com/user-attachments/assets/e2a7a5e2-651a-4b0d-b3f9-12cb40acce04)

## Installation 💻

### Option 1: ComfyUI Manager 📦
The easiest way to install is through [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager):
1. Install ComfyUI Manager if you haven't already
2. Open ComfyUI
3. Navigate to the Manager tab
4. Search for "Control Nodes"
5. Click Install

### Option 2: Manual Installation ⚙️
1. Clone this repository into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ryanontheinside/ComfyUI_RealTimeNodes
```

2. Install the required dependencies:
```bash
cd ComfyUI_RealTimeNodes
pip install -r requirements.txt
```

## Coming Soon 🚀

### Advanced Detection System 👁️
The next major update will introduce a comprehensive detection system with multiple detector types:
- **Motion Detection**: Enhanced motion detection with configurable sensitivity and regions
- **Object Detection**: Real-time object detection and tracking
- **Face Detection**: Face detection and landmark tracking
- **Pose Detection**: Human pose estimation and tracking
- **Depth Detection**: Real-time depth estimation and segmentation

These detection systems will provide powerful inputs for real-time ComfyUI workflows, enabling dynamic responses to various types of detected changes in the input stream.

## Project Status 📊

This is an evolving project that aims to expand the real-time capabilities of ComfyUI. As real-time use cases for ComfyUI continue to emerge and grow, this project will adapt and expand to meet those needs. The goal is to provide a comprehensive suite of tools for real-time workflows, from simple value animations to complex detection and response systems.

### Contributing 🤝

Your feedback and contributions are more than welcome! This project grows stronger with community input.

- Have an idea? Open an issue! 💡
- Found a bug? Open an issue! 🐛
- Made an improvement? Submit a PR! 🎉
- Want to help? Join the discussion! 💬

Please visit our [GitHub Issues](https://github.com/ryanontheinside/ComfyUI_RealTimeNodes/issues) page to contribute.

## Related Projects 🔗

### ComfyUI_RyanOnTheInside - Everything Reactivity ⚡
Make anything react to anything in your ComfyUI workflows. [ComfyUI_RyanOnTheInside](https://github.com/ryanontheinside/ComfyUI_RyanOnTheInside) - my main custom nodes suite that brings complete reactive control to standard ComfyUI workflows:

- Dynamic node relationships
- React to audio, MIDI, motion, time, depth, color, Whisper, and more
- Audio source separation and manipulation
- Reactive particle systems
- Reactive text generation
- Reactive image generation
- Reactive video generation
- Optical flow
- Reactive IPAdapters and CogVideo
- Reactive Live Portrait
- Reactive DepthFlow
- Actually more

Use it alongside these Control Nodes to master parameter control in both the batch and real-time paradigms in ComfyUI! The POWER!!