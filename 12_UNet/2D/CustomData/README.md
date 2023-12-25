# UNet with Padding Implementation

This repository contains an implementation of a UNet architecture **with** padding, which deviates from the architecture in the original paper for two dirrefent data sets: Carvana and DRIVE database for segmentation of blood vessels in retinal images. Please note that the data preprocessing pipeline, model architecture, and training are the same for both completely different datasets, which reflects the robustness of the presented implementation. The key feature of this implementation is a unified feature map shape, along with consistent input and output image dimensions. This design choice aims to simplify the network architecture and enhance overall clarity, reducing coding complications.

Below is an example of semantic segmentation performed by the implemented network on the Carvana dataset after a single epoch. Remarkably, even with just one epoch of training, the quality of the segmentation results is impressive.

Ground truth:
![1](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/142d7497-d5eb-4a3f-8b50-c25f93e4d817)
Model's output:
![pred_1](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/1cbfc970-c786-4292-9424-ace3d5866520)

Below is an example of semantic segmentation conducted by the implemented network on the DRIVE dataset after 30 epochs. Notably, the model excels in segmenting thicker blood vessels, but encounters difficulty in detecting smaller ones, treating them as image noise. This challenge may arise from the low resolution of the input images.
![combined_image](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/8b38d857-7d87-4448-be42-d3fe1107f540)
![0](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/04dc23fe-2423-464a-9383-0bcb840461d6)
![pred_0](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/e04cc6ff-5a18-4b5e-9947-53587e8ba467)





