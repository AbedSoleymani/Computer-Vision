# UNet for Image Segmentation

## Overview

Explore the UNet architecture for image segmentation in this repository. UNet, a fully convolutional neural network (FCN), is commonly used for semantic segmentation tasks.
UNet is widely used in a variety of image segmentation tasks, such as medical image segmentation, satellite image analysis, and object detection in autonomous vehicles. It is known for its ability to handle high-resolution images and to produce accurate segmentation maps.
Here we will provide a brief theoretical insights into UNet and addresse common questions about its architecture and applications.
<div align="center">
<img width="684" alt="image" src="https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/c293af67-a715-44f5-b7d7-964a07eb86c6">
<div align="left">

## Table of Contents

- [UNet Architecture](#UNet-architecture)
- [Training](#training)
- [FAQs](#faqs)

## UNet Architecture
UNet is a fully convolutional neural network architecture (with no fully connected layers) designed for semantic segmentation. It comprises a contracting path to capture context, a symmetric expanding path for precise localization, and skip connections to retain fine-grained information.
* **Contracting Path:** The initial layers of the network contract the input's spatial dimensions, capturing high-level contextual information through convolutional and pooling operations.
* **Expanding Path:** A symmetric expanding path reconstructs spatial details, upsampling the feature maps to generate a precise segmentation map.
* **Skip Connections:** Skip connections connect one or several layers of the contracting path to the corresponding parts of the expanding path according to the network's depth. These connections allow the network to retain fine-grained details, addressing information loss during downsampling and upsampling, increasing the segmentation map’s precision. As shown in the following figure, the UNet can produce acceptable segmentation even without skip connections, but the added skip connections can introduce finer details (see the join between the two ellipses on the right).

<div align="center">
<img width="1281" alt="image" src="https://github.com/AbedSoleymani/Computer-Vision/assets/72225265/a4f14573-f2af-452c-b570-b93c42f08edd">
<div align="left">

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
A: In UNet, due to the use of unpadded convolutions, the output image size is smaller than the input by a constant border width. This results in a difference in dimensions between corresponding layers in the contracting and expanding paths. When forming skip connections, the dimensions are intentionally mismatched. The model handles this by cropping the feature maps from the contracting path to match the dimensions of the corresponding feature maps in the expanding path. The skip connection is then performed by concatenating these adjusted feature maps. Even though no justification were given for not using padding and keeping the sahpes identical in contracting and expanding paths, it seems it was because the authors didn’t want to introduce segmentation errors at the image margin!
For an image with input size $572 \times 572$, the output will be $388 \times 388$, i.e., $1-\frac{388 \times 388}{572 \times 572}=0.54$ or a $\approx 50$\% loss! If we want to run UNet without padding, you need to run it multiple times on overlapping tiles to get the full segmentation image.

### Q: Advantages and Disadvantages of Using the UNet Architecture for Image Segmentation Tasks.
A:

* Advantages:

1. High performance: UNet is known for producing accurate segmentation maps, particularly when working with high-resolution images or datasets with many classes.

2. Good handling of multi-class tasks: UNet is well-suited for multi-class image segmentation tasks, as it can handle a large number of classes and produce a pixel-level segmentation map for each class.

3. Efficient use of training data: UNet uses skip connections, which allow the model to incorporate high-level and low-level features from the input image. This can make UNet more efficient at using the training data and improve the model’s performance.

* Disadvantages:

1. A large number of parameters: UNet has many parameters due to the skip connections and the additional layers in the expanding path. This can make the model more prone to overfitting, especially when working with small datasets.

2. High computational cost: UNet requires additional computations due to the skip connections, which can make it more computationally expensive than other architectures.

### Q: Handling Multi-class Image Segmentation: Challenges
A: The UNet architecture is well-suited for handling multi-class image segmentation tasks, as it can produce a pixel-level segmentation map for each class. In a multi-class image segmentation task, the UNet model is trained on a large dataset of annotated images, where each pixel is labeled with the class to which it belongs. The model is then used to predict the class label for each pixel in a new image.

One challenge of multi-class image segmentation is the imbalanced distribution of classes in the training data. For example, if multiple classes of objects exist in the image, some classes may be much more common than others. This can lead to bias in the model, as it may be more accurate at predicting the more common classes and less accurate at predicting the less common classes. To address this issue, it may be necessary to balance the training data by oversampling the less common classes or using data augmentation techniques.

Another challenge of multi-class image segmentation is handling class overlap, where pixels may belong to multiple classes. For example, in a medical image segmentation task, the boundary between two organs may be difficult to distinguish, as the pixels in this region may belong to both organs. To address this issue, it may be necessary to use a model capable of producing a probabilistic segmentation map, where each pixel is assigned a probability of belonging to each class.
