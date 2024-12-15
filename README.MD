# ComfyUI Control Nodes

Simple control nodes that generate changing values over separate executions of a workflow. Each time the workflow is executed, the value will be updated. This is useful for animating parameters in systems like ComfyStream that execute the workflow for each frame.

## Nodes

### FloatControl
Outputs a floating point value that changes over time using various patterns (sine wave, bounce, random walk, etc). 

### IntControl 
Same as FloatControl but outputs integer values.


## Usage
Connect to the int or float input of a node you want to animate.

IMPORTANT: be sure to set viable minimum and maximum values according to the node you are animating.


More soon!