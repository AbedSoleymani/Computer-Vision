"""
This is the implementation of UNet with padded convolutions.
In contrast to the original paper, the shape of input and output images as well as
all of the middle feature maps are compatible and we do not need to crop feature
maps for skip connection. This will simplify the implementation and functionality.
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1, # to make it the same convolution
                      bias=False, # Because we want to use BN in the next layer
                      ),
            nn.BatchNorm2d(out_channels),

            # When `inplace`  in `ReLU` is set to `True`, it means that the output tensor will be written
            # to the same memory location as the input tensor, overwriting the values in the input tensor.
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, # The second conv in the block does not change the depth
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1, # to make it the same convolution
                      bias=False, # Because we want to use BN in the next layer
                      ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)
    

