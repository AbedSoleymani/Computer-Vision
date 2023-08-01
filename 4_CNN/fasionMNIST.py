import torch
import torchvision

from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

class Data():
    def __init__(self, batch_size=20):
        self.batch_size = batch_size

    def generate(self):
        # The output of torchvision datasets are PILImage images of range [0, 1]. 
        # We transform them to Tensors for input into a CNN
        data_transform = transforms.ToTensor()

        train_data = FashionMNIST(root='./data',
                                  train=True,
                                  download=True,
                                  transform=data_transform)
        
        test_data = FashionMNIST(root='./data',
                                 train=False,
                                 download=True,
                                 transform=data_transform)

        print('Train data, number of images: ', len(train_data))
        print('Test data, number of images: ', len(test_data))

        train_loader = DataLoader(train_data,
                                  batch_size=self.batch_size,
                                  shuffle=True)
        
        test_loader = DataLoader(test_data,
                                 batch_size=self.batch_size,
                                 shuffle=True)

        classes = ['T-shirt/top',
                   'Trouser',
                   'Pullover',
                   'Dress',
                   'Coat', 
                   'Sandal',
                   'Shirt',
                   'Sneaker',
                   'Bag',
                   'Ankle boot']
        return train_loader, test_loader, classes
    
    def visualize(self, data_generator, classes):
        dataiter = iter(data_generator)
        images, labels = next(dataiter)
        images = images.numpy()

        fig = plt.figure(figsize=(25, 4))
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(2, self.batch_size//2, idx+1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(images[idx]), cmap='gray')
            ax.set_title(classes[labels[idx]])