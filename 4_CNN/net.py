import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self, weight):
        super(Net, self).__init__()
        k_height, k_width = weight.shape[2:]
        self.conv = nn.Conv2d(in_channels=1, # for gray image
                              out_channels=weight.shape[0],
                              kernel_size=(k_height,
                                           k_width),
                              bias=False)
        self.conv.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        conv_output = self.conv(x)
        activatio_output = F.relu(conv_output)

        return conv_output, activatio_output
    
    def viz_layer(self, layer, n_filters= 4):
        fig = plt.figure(figsize=(20, 20))
        
        for i in range(n_filters):
            ax = fig.add_subplot(1,
                                 n_filters,
                                 i+1,
                                 xticks=[],
                                 yticks=[])
            # grab layer outputs
            ax.imshow(np.squeeze(layer[0,i].data.numpy()),
                      cmap='gray')
            ax.set_title('Output %s' % str(i+1))