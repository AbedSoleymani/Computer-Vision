# UNET for Image Segmentation

## Overview

Explore the UNET architecture for image segmentation in this repository. UNET, a fully convolutional neural network (FCN), is commonly used for semantic segmentation tasks. Here we will provide a brief theoretical insights into UNET and addresse common questions about its architecture and applications.

## Table of Contents

- [UNET Architecture](#unet-architecture)
- [Training](#training)
- [FAQs](#faqs)

## UNET Architecture
UNET is a fully convolutional neural network architecture (with no fully connected layers) designed for semantic segmentation. It comprises a contracting path to capture context, a symmetric expanding path for precise localization, and skip connections to retain fine-grained information.
* **Contracting Path:** The initial layers of the network contract the input's spatial dimensions, capturing high-level contextual information through convolutional and pooling operations.
* **Expanding Path:** A symmetric expanding path reconstructs spatial details, upsampling the feature maps to generate a precise segmentation map.
* **Skip Connections:** Skip connections connect several levels of the contracting path to the corresponding parts of the expanding path according to the network's depth. These connections allow the network to retain fine-grained details, addressing information loss during downsampling and upsampling.

## Training
The network is trained using input images paired with their respective segmentation maps through stochastic gradient descent implementation. It's essential to highlight that, as illustrated in the above image, the output depth is configured to 2. This setting aligns with having two classes for segmentation— one for the object of interest and one for the background. If the segmentation task involves more classes, the depth of the output needs adjustment by changing the final number of convolution levels accordingly.

Due to the unpadded convolutions, the output image is smaller than the input by a constant border width.

To minimize the overhead and make maximum use of the GPU memory, we favor large input tiles over a large batch size and hence **reduce the batch to a single image**. Accordingly we use a **high momentum** (0.99) such that a large number of the previously seen training samples determine the update in the current optimization step.

**Data augmentation** is essential to teach the network the desired invariance and robustness properties, when only few training samples are available. In case of medical images, we primarily need shift and rotation invariance as well as robustness to deformations and gray value variations. Especially random elas- tic deformations of the training samples seem to be the key concept to train a segmentation network with very few annotated images. We generate smooth deformations using random displacement vectors on a coarse 3 by 3 grid. The displacements are sampled from a Gaussian distribution with 10 pixels standard deviation. Per-pixel displacements are then computed using bicubic interpola- tion. Drop-out layers at the end of the contracting path perform further implicit data augmentation.

## FAQs
