# UNET for Image Segmentation

## Overview

Explore the UNET architecture for image segmentation in this repository. UNET, a fully convolutional neural network (FCN), is commonly used for semantic segmentation tasks. Here we will provide a brief theoretical insights into UNET and addresse common questions about its architecture and applications.

## Table of Contents

- [UNET Architecture](#unet-architecture)
- [Training](#training)
- [Applications](#applications)
- [FAQs](#faqs)

## UNET Architecture
UNET is a fully convolutional neural network architecture (with no fully connected layers) designed for semantic segmentation. It comprises a contracting path to capture context, a symmetric expanding path for precise localization, and skip connections to retain fine-grained information.
* Contracting Path: The initial layers of the network contract the input's spatial dimensions, capturing high-level contextual information through convolutional and pooling operations.
* Expanding Path: A symmetric expanding path reconstructs spatial details, upsampling the feature maps to generate a precise segmentation map.
* Skip Connections: Skip connections connect several levels of the contracting path to the corresponding parts of the expanding path according to the network's depth. These connections allow the network to retain fine-grained details, addressing information loss during downsampling and upsampling.