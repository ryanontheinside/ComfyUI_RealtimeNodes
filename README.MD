# ComfyUI Control Nodes

Simple control nodes that generate changing values over separate executions of a workflow. Each time the workflow is executed, the value will be updated. This is useful for animating parameters in systems like ComfyStream that execute the workflow once per frame frame.

## Nodes

### FloatControl
Outputs a floating point value that changes over time using various patterns (sine wave, bounce, random walk, etc). This

### IntControl 
Same as FloatControl but outputs integer values.

### StringControl
Same as the other control nodes, but cycles through a list of strings.

## Usage
Connect the control node to the input of the node you want to animate.

IMPORTANT: be sure to set viable minimum and maximum values according to the node you are animating.

More soon!

## Demo
In this  demo we are controlling the width and height of a shape mask with an Int Control node. Imagine controlling the denoise on a KSampler with a Float Control, though!
![Value Control Demo](assets/ValueControl.gif)