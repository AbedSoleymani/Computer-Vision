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
    

class UNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 features=[64, 128, 256, 512]):
        
        super(UNet, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.expanding_path = nn.ModuleList()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path of the UNet
        for feature in features:
            self.contracting_path.append(DoubleConv(in_channels=in_channels,
                                                    out_channels=feature))
            in_channels = feature

        # Expanding path of the UNet
        for feature in reversed(features):
            self.expanding_path.append(nn.ConvTranspose2d(in_channels=feature*2,
                                                          out_channels=feature,
                                                          kernel_size=2,
                                                          stride=2))
            self.expanding_path.append(DoubleConv(in_channels=feature*2,
                                                  out_channels=feature))
            
        self.bottleneck = DoubleConv(in_channels=features[-1],
                                     out_channels=features[-1]*2)
        
        self.final_conv = nn.Conv2d(in_channels=features[0],
                                    out_channels=out_channels,
                                    kernel_size=1)
        
    def forward(self, x):
        self.skip_connections = []
        
        for down_block in self.contracting_path:
            x = down_block(x)
            self.skip_connections.append(x)
            x = self.pooling(x)

        x = self.bottleneck(x)

        # While creating `self.expanding_path`, we appened two processing to the input
        # That is why we use step of two in the next line of code.
        for idx in range(0, len(self.expanding_path), 2):
            x = self.expanding_path[idx](x)
            skip_connection = self.skip_connections[idx//2]

            # The next two lines are for the case that the input image shape is not perfectly 
            # devisible by 16 and it will make a size mismatch while concatenating x and skip_connection
            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.expanding_path[idx+1](concat_skip)

        return self.final_conv(x)