import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

def show_rand_img(dataset):
    rand_i = np.random.randint(0, len(dataset))
    sample = dataset[rand_i]
    show_keypoints(sample['image'], sample['keypoints'])

def show_keypoints(image, key_pts):
    """Shows a given image with its corresponding keypoints"""
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')
    plt.show()