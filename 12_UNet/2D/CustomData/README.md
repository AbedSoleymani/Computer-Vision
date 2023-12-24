# UNet with Padding Implementation

This repository contains an implementation of a UNet architecture **with** padding, which deviates from the architecture in the original paper. The key feature of this implementation is a unified feature map shape, along with consistent input and output image dimensions. This design choice aims to simplify the network architecture and enhance overall clarity, reducing coding complications.

Below is an example of semantic segmentation performed by the implemented network on the Carvana dataset after a single epoch. Remarkably, even with just one epoch of training, the quality of the segmentation results is impressive.