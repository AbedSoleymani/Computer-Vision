# (Vanilla/Res/Attention)UNet with Padding Implementation

This repository contains an implementation of a UNet architecture **with** padding, which deviates from the architecture in the original paper for two dirrefent data sets: Carvana and DRIVE database for segmentation of blood vessels in retinal images. Please note that the data preprocessing pipeline, model architecture, and training are the same for both completely different datasets, which reflects the robustness of the presented implementation. The key feature of this implementation is a unified feature map shape, along with consistent input and output image dimensions. This design choice aims to simplify the network architecture and enhance overall clarity, reducing coding complications.

Below is an example of semantic segmentation performed by the implemented network on the Carvana dataset after a single epoch. Remarkably, even with just one epoch of training, the quality of the segmentation results is impressive.

Ground truth:
![1](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/142d7497-d5eb-4a3f-8b50-c25f93e4d817)
Model's output:
![pred_1](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/1cbfc970-c786-4292-9424-ace3d5866520)

Below is an example of semantic segmentation performed by the implemented network on the DRIVE dataset after $30$ epochs for `512x512` images. Remarkably, the model excels at segmenting thicker blood vessels but faces challenges in detecting tiny ones, treating them as image noise. This difficulty may arise from factors such as the low resolution of the image, the small size of the training set (20 samples), the potential loss of tiny features in the contracting path, or the shallowness of the model.
![combined_image](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/8b38d857-7d87-4448-be42-d3fe1107f540)
![0](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/04dc23fe-2423-464a-9383-0bcb840461d6)
![pred_0](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/e04cc6ff-5a18-4b5e-9947-53587e8ba467)

Below is an example of semantic segmentation performed by the ResUNet on the DRIVE dataset. To convert the original UNet architecture to its residual version, set the `residual` variable to `True` as follows:
```python
model = UNet(in_channels=3, out_channels=1, residual=True).to(device)
```
It was observed that ResUNet achieved a slightly better Dice score compared to the vanilla one after $30$ epochs for the same `256x256` input images. The following images showcase examples of the model's performance on `256x256` images:
![combined_image](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/ad92090b-e662-43a6-8aea-b2b6ff48849c)
![0](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/25cd4863-57e6-4893-ad0b-69d9046dc0b7)
![pred_0](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/e69b50b0-92b7-45f9-b46e-216fa28db5ab)

As shown in the images, due to the lower resolution, the quality of details in `256x256` images is lower than in `512x512` inputs, despite the slightly higher Dice score (Dice scores: $0.68$ and $0.63$, respectively).

## Attention UNet
Advantages of Attention UNet over the vanilla UNet lie in its ability to enhance model sensitivity and prediction accuracy without significant computational overhead. By integrating Attention Gates (AGs) into the standard UNet architecture, the model gains the capability to highlight relevant activations during training. This strategic attention mechanism reduces computational resources wasted on irrelevant activations, leading to improved overall performance.

The integration of attention in UNet can be implemented in terms of two attention types: Hard Attention and Soft Attention. Hard Attention focuses on cropping relevant regions, addressing one region at a time through non-differentiable reinforcement learning. On the other hand, Soft Attention involves weighting different parts of the image, assigning larger weights to relevant areas and smaller weights to less relevant regions. Soft Attention can be trained with backpropagation, allowing the model to dynamically adjust weights during training, emphasizing the significance of relevant regions. In this repository, Soft Attention is employed to dynamically assign weights to different parts of the image during the training of the UNet model, leveraging the ease of training with backpropagation.

Adding attention to UNet can be an important improvement as it combines spatial information from the contracting path with the expanding path to retain valuable spatial details. While this process is beneficial, it also introduces richer feature representation from the initial layers. Soft attention, when implemented at skip connections, actively suppresses activations at irrelevant regions (i.e., spacial information from the background), mitigating the impact of poor feature representation.

The structure of the attention block further enhances the model's capabilities. The Attention Gate takes two inputs: $x$, from skip connections with better spatial information, and $g$, the gating signal from the next lowest layer with superior feature representation. This dual-input mechanism ensures that the model leverages both early-layer spatial information and deep-layer feature representation for more effective and context-aware predictions.
![Unknown-9](https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/bb3b83d4-2e62-4399-af11-5cddd441a410)
The attention heat map displayed above provides a visual representation of the performance of the Attention UNet model on the Caravana dataset after just two epochs of training.
The map highlights regions with elevated values over the car, indicating the model's effective focus on relevant features during the segmentation process. This concentration of attention on the car pixels, accompanied by lower values in the background, signifies the model's ability to discern and prioritize crucial details for accurate segmentation.
