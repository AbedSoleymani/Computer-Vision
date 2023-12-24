# UNet with Padding Implementation

This repository contains an implementation of a UNet architecture **with** padding, which deviates from the architecture in the original paper. The key feature of this implementation is a unified feature map shape, along with consistent input and output image dimensions. This design choice aims to simplify the network architecture and enhance overall clarity, reducing coding complications.

Below is an example of semantic segmentation performed by the implemented network on the Carvana dataset after a single epoch. Remarkably, even with just one epoch of training, the quality of the segmentation results is impressive.

Ground truth:
![1](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/142d7497-d5eb-4a3f-8b50-c25f93e4d817)
Model's output:
![pred_1](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/1cbfc970-c786-4292-9424-ace3d5866520)
