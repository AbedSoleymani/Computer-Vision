# UNet for Image Segmentation

## Overview

Explore the UNet architecture for image segmentation in this repository. UNet, a fully convolutional neural network (FCN), is commonly used for semantic segmentation tasks.
UNet is widely used in a variety of image segmentation tasks, such as medical image segmentation, satellite image analysis, and object detection in autonomous vehicles. It is known for its ability to handle high-resolution images and to produce accurate segmentation maps.
Here we will provide a brief theoretical insights into UNet and addresse common questions about its architecture and applications.

## Table of Contents

- [UNet Architecture](#UNet-architecture)
- [Training](#training)
- [FAQs](#faqs)

## UNet Architecture
UNet is a fully convolutional neural network architecture (with no fully connected layers) designed for semantic segmentation. It comprises a contracting path to capture context, a symmetric expanding path for precise localization, and skip connections to retain fine-grained information.
* **Contracting Path:** The initial layers of the network contract the input's spatial dimensions, capturing high-level contextual information through convolutional and pooling operations.
* **Expanding Path:** A symmetric expanding path reconstructs spatial details, upsampling the feature maps to generate a precise segmentation map.
* **Skip Connections:** Skip connections connect one or several layers of the contracting path to the corresponding parts of the expanding path according to the network's depth. These connections allow the network to retain fine-grained details, addressing information loss during downsampling and upsampling, increasing the segmentation map’s precision..

## Training
The network is trained using input images paired with their respective segmentation maps through stochastic gradient descent implementation. It's essential to highlight that, as illustrated in the above image, the output depth is configured to 2. This setting aligns with having two classes for segmentation— one for the object of interest and one for the background. If the segmentation task involves more classes, the depth of the output needs adjustment by changing the number of convolution at the final layer accordingly.

Due to the unpadded convolutions, the output image is smaller than the input by a constant border width.

To minimize the overhead and make maximum use of the GPU memory, we favor large input tiles over a large batch size and hence **reduce the batch to a single image**. Accordingly we use a **high momentum** (0.99) such that a large number of the previously seen training samples determine the update in the current optimization step.

**Data augmentation** is essential to teach the network the desired invariance and robustness properties, when only few training samples are available. In case of medical images, we primarily need shift and rotation invariance as well as robustness to deformations and gray value variations. Especially random elas- tic deformations of the training samples seem to be the key concept to train a segmentation network with very few annotated images. We generate smooth deformations using random displacement vectors on a coarse 3 by 3 grid. The displacements are sampled from a Gaussian distribution with 10 pixels standard deviation. Per-pixel displacements are then computed using bicubic interpola- tion. Drop-out layers at the end of the contracting path perform further implicit data augmentation.

## FAQs
### Q: Why not use a flat architecture with 3x3 convolutions and same padding, having feature maps with the same size as the input image?
A: While a flat architecture with same padding and equal-sized feature maps is a valid approach, namely "Fully Convolutional Networks (FCN)". The key distinctions between the UNet architecture and FCN lie in their structural design, use of skip connections, number of parameters, computational efficiency, and performance characteristics.
1. **Architecture:**
UNet introduces the U-shaped model for semantic segmentation. The U-shaped architecture, with its contracting and expanding paths, offers several advantages over a flat design. The contracting path with numerous 2x2 max pooling operation with stride 2 for downsampling favors a larger receptive field, enhancing contextual understanding, hierarchical feature learning, and precise localization—critical aspects for addressing the challenges of segmentation tasks.
FCN adopts a single encoder-decoder structure. The encoder includes convolutional and max pooling layers for downsampling and feature extraction, and the decoder consists of convolutional and upsampling layers for upsampling feature maps and producing the segmentation map. Please note that, there is no skip connections.
2. **Number of Parameters:**
Typically, UNet has more parameters than FCN due to the inclusion of skip connections and additional layers in the expanding path. This may make UNet more susceptible to overfitting, especially when dealing with smaller datasets.
FCN tends to have fewer parameters compared to UNet, contributing to potentially lower overfitting risks, which can be advantageous when working with limited data.
3. **Computational Efficiency:**
The presence of skip connections and additional computations for feature fusion may make UNet computationally more intensive compared to FCN. This could impact tasks requiring fast inference times or limited computational resources.
FCN is generally more computationally efficient than UNet due to its simpler structure and lower parameter count. This efficiency can be beneficial in scenarios where computational resources are a consideration.
4. **Performance:**
General Performance: In general, UNet tends to outperform FCN on image segmentation tasks, particularly in scenarios involving high-resolution images or datasets with a large number of classes.

Task Dependence: The performance of both architectures can vary based on the specific segmentation task and the quality and quantity of training data. UNet's hierarchical feature fusion and skip connections contribute to its effectiveness in tasks requiring detailed segmentation.

### Q: In UNet, how does the model perform skip connections when the dimensions of corresponding layers in the contracting and expanding paths are not the same?
A: In UNet, due to the use of unpadded convolutions, the output image size is smaller than the input by a constant border width. This results in a difference in dimensions between corresponding layers in the contracting and expanding paths. When forming skip connections, the dimensions are intentionally mismatched. The model handles this by cropping the feature maps from the contracting path to match the dimensions of the corresponding feature maps in the expanding path. The skip connection is then performed by concatenating these adjusted feature maps.

