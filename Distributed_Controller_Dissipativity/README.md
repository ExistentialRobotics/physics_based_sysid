# Dissipativity Leveraged for Distributed Control
This repo provides code for Example 3.2.

## Dependencies
This code is developed and tested in MATLAB. The following MATLAB toolboxes and external libraries are required:

```MATLAB Optimization Toolbox```: https://www.mathworks.com/products/optimization.html.

``` YAlMIP ```: Download at [https://yalmip.github.io/](https://yalmip.github.io/download/) and 
add YALMIP to your MATLAB path. <pre> ```addpath(genpath('path_to_YALMIP_folder')) ``` </pre>
```BMIBNB```: Built-in solver used by YALMIP for nonconvex problems.


## Demo with Three Second Order Subsystems
Run ```distributed_controller.m``` to design distributed linear controllers for three subsystems that ensure suitable dissipativity conditions.
We simulate and plot the closed-loop dynamics with noise and coupling to demonstrate the performance for the controllers. 
