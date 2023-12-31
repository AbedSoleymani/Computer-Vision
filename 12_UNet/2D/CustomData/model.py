"""
This is the implementation of UNet with padded convolutions.
In contrast to the original paper, the shape of input and output images as well as
all of the middle feature maps are compatible and we do not need to crop feature
maps for skip connection. This will simplify the implementation and functionality.
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

# device = "cuda" if torch.cuda.is_available() else "cpu" # for Google Colab
device = "mps" if torch.backends.mps.is_available() else "cpu" # for Apple Silicon

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        self.residual = residual
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        residual = out
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.residual:
            return out + residual
        else:
            return out
    

class UNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 features=[64, 128, 256, 512],
                 residual=False,
                 attention=False):
        
        self.residual = residual
        self.attention = attention

        super(UNet, self).__init__()

        self.contracting_path = nn.ModuleList()
        self.expanding_path = nn.ModuleList()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path of the UNet
        for feature in features:
            self.contracting_path.append(DoubleConv(in_channels=in_channels,
                                                    out_channels=feature,
                                                    residual=self.residual))
            in_channels = feature

        # Expanding path of the UNet
        for feature in reversed(features):
            self.expanding_path.append(nn.ConvTranspose2d(in_channels=feature*2,
                                                          out_channels=feature,
                                                          kernel_size=2,
                                                          stride=2))
            self.expanding_path.append(DoubleConv(in_channels=feature*2,
                                                  out_channels=feature,
                                                  residual=self.residual))
            
        self.bottleneck = DoubleConv(in_channels=features[-1],
                                     out_channels=features[-1]*2,
                                     residual=self.residual)
        
        self.final_conv = nn.Conv2d(in_channels=features[0],
                                    out_channels=out_channels,
                                    kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        for down_block in self.contracting_path:
            x = down_block(x)
            skip_connections.append(x)
            x = self.pooling(x)

        x = self.bottleneck(x)

        # Now, we reverse the `skip_connection` list since we are going from down to up in expanding path.
        skip_connections = skip_connections[::-1]

        # While creating `self.expanding_path`, we appened two processing to the input
        # That is why we use step of two in the next line of code.
        for idx in range(0, len(self.expanding_path), 2):
            x = self.expanding_path[idx](x)
            skip_connection = skip_connections[idx//2]

            # The next two lines are for the case that the input image shape is not perfectly 
            # devisible by 16 and it will make a size mismatch while concatenating x and skip_connection
            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])
            
            if self.attention:
                attention_model = Attention(gate=x, skip_connection=skip_connection).to(device)
                attention_map = attention_model()
                attentive_skip = torch.mul(attention_map, skip_connection)
                self.final_attection_map = attention_map # for visualization
                output = torch.cat((attentive_skip, x), dim=1)
            else:
                output = torch.cat((skip_connection, x), dim=1)
            x = self.expanding_path[idx+1](output)

        return self.final_conv(x)
    
class Attention(nn.Module):
    def __init__(self, gate, skip_connection):
        self.gate = gate
        self.skip_connection = skip_connection
        self.num_channels = gate.shape[1] # in this srchitecture, skip_connection and gate have the same shape

        super(Attention, self).__init__()

        self.atten_conv1 = nn.Conv2d(in_channels=self.num_channels,
                                    out_channels=self.num_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.atten_conv2 = nn.Conv2d(in_channels=self.num_channels,
                                    out_channels=1, # to creat the attention map
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=False)
        self.bn2 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self):

        g = self.bn1(self.atten_conv1(self.gate))
        x = self.bn1(self.atten_conv1(self.skip_connection))

        psi = self.relu(g + x)
        sigma = self.bn2(self.atten_conv2(psi))

        attention_map = self.sigmoid(sigma)

        return attention_map


def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNet(in_channels=1, out_channels=1, attention=True)
    preds = model(x)
    print (preds.shape)
    print (x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()