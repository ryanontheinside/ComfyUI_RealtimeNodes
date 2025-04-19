# ComfyUI Real-Time Nodes ⏱️

A growing suite of nodes for real-time ComfyUI workflows. Features include value animation, motion detection and tracking, sequence control, and more. These nodes update their outputs on each workflow execution, making them perfect for real-time applications like [ComfyStream](https://github.com/yondonfu/comfystream) that execute the workflow once per frame. 

The intention for this repository is to build a suite of nodes that can be used in the burgeoning real-time diffusion space. Contributions are welcome!


## Control Nodes

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
- **SimilarityFilter**: Filter out similar consecutive images and control downstream execution. Perfect for optimizing real-time workflows by skipping redundant processing of similar frames.

### Logic 🧠
- **LazyCondition**: Powerful conditional execution node that supports any input type. Uses lazy evaluation to truly skip execution of unused paths and maintains state to avoid feedback loops.



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

### Utility Controls
Use utility nodes to optimize and control your workflow:
- **FPS Monitor**: Monitor performance with a visual overlay
- **SimilarityFilter**: Skip processing of similar frames by comparing consecutive images. Great for optimizing real-time workflows by only processing frames that have meaningful changes.
- **LazyCondition**: Create conditional execution paths that truly skip processing of unused branches. Works with any input type (images, latents, text, numbers) and maintains state of the last successful output to avoid feedback loops.

## 🔮 MediaPipe Vision

### ✨ Overview

This repository provides a complete implementation of Google MediaPipe vision tasks for ComfyUI. It enables computer vision capabilities that can be used for interactive AI art, responsive interfaces, motion tracking, and advanced masking workflows.

### 🚀 Features

| Category | Available Tools |
|----------|-------------|
| **Face Analysis** | Face detection, face mesh (478 points), blendshapes, head pose |
| **Body Tracking** | Pose estimation (33 landmarks), segmentation masks |
| **Hand Analysis** | Hand tracking (21 landmarks per hand), gesture recognition |
| **Image Processing** | Object detection, image segmentation, image embeddings |
| **Creative Tools** | Face stylization, interactive segmentation |

### 📋 Supported MediaPipe Tasks

* **Face Detection:** Face bounding boxes and keypoints
* **Face Landmark Detection:** Face mesh landmarks with expression analysis
* **Hand Landmark Detection:** Hand position tracking with 21 landmarks
* **Pose Landmark Detection:** Body pose tracking with 33 landmarks
* **Object Detection:** Common object detection using models like EfficientDet
* **Image Segmentation:** Category-based image segmentation
* **Gesture Recognition:** Recognition of common hand gestures
* **Image Embedding:** Feature vector generation for image similarity
* **Interactive Segmentation:** User-guided image masking
* **Face Stylization:** Artistic style application to faces
* **Holistic Landmark Detection:** Full-body landmark detection (legacy)

> **Note:** Holistic landmark detection uses the legacy MediaPipe API as we await the official Tasks API release.

### ⚙️ Landmark System

The project's landmark system allows extracting and using position data:

#### Position Extraction

**Landmark Position Extractors** access coordinate data from any landmark:
- Extract x, y, z positions from face, hand, or pose landmarks
- Access visibility and presence information where available
- Access world coordinates when available (hand and pose)
- Input landmark indices directly to access any point
- Process batches for multi-frame workflows

#### Position Processing

Several node types work with landmark position data:

- **Delta Controls** - Track movement and map changes to parameter values
- **Proximity Nodes** - Calculate distances between landmarks
- **Masking Nodes** - Generate masks centered at landmark positions
- **Head Pose Extraction** - Calculate yaw, pitch, roll from face landmarks
- **Blendshape Analysis** - Extract facial expression parameters

### Example Workflow

```
Load Face Landmarker → Face Landmarker ← Image Input
           |
           ↓ landmarks
Face Landmark Position (Index: 1) → x,y,z coordinates
           |
           ↓ x,y,z
Position Delta Float Control → value → ComfyUI Parameter
```


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
> **Note:** For MediaPipe, GPU Support varies by platform. For Linux, see [these instructions](https://ai.google.dev/edge/mediapipe/framework/getting_started/gpu_support).

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

This project provides flexible infrastructure for computer vision in ComfyUI. If you have ideas for:

- Creative AI interactions using vision
- Specific landmark tracking or detection needs
- Real-time vision workflows
- Improvements to the current implementation

Please open an issue, even if you're not sure how to implement it. 

The aim is to **iterate quickly** to keep up with this burgeoning field of real-time ComfyUI 


Please visit our [GitHub Issues](https://github.com/ryanontheinside/ComfyUI_RealTimeNodes/issues) page to contribute.

## Related Projects 🔗

## 🔗 Related Projects

### [ComfyUI_ControlFreak](https://github.com/ryanontheinside/ComfyUI_ControlFreak)
Universal MIDI & Gamepad Mapping in ComfyUI. Map any MIDI controller or gamepad to any parameter in your ComfyUI workflow for intuitive, hands-on control of your generative art. Perfect for live performances, interactive installations, and streamlined creative workflows.

### [comfystream](https://github.com/yondonfu/comfystream)
A real-time streaming framework for ComfyUI that enables running workflows continuously on video streams, perfect for 
combining with MediaPipe vision capabilities.

### [ComfyUI-Stream-Pack](https://github.com/livepeer/ComfyUI-Stream-Pack)
A collection of ComfyUI nodes for multimedia streaming applications. Combines video processing with generative models 
for real-time media effects.

### [ComfyUI_RyanOnTheInside](https://github.com/ryanontheinside/ComfyUI_RyanOnTheInside) - Everything Reactivity ⚡
Make anything react to anything in your ComfyUI workflows.  - my main custom nodes suite that brings complete reactive control to standard ComfyUI workflows:

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

