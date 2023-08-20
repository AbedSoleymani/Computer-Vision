import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from preprocessing import Rescale, RandomCrop, FaceCrop, Normalize, ToTensor, Random90DegFlip
from torchvision import transforms

def show_rand_img(dataset):
    rand_i = np.random.randint(0, len(dataset))
    sample = dataset[rand_i]
    show_keypoints(sample['image'], sample['keypoints'])
    plt.show()


def show_keypoints(image, key_pts):
    """Shows a given image with its corresponding keypoints"""
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')


def show_preprocessing_results(dataset, test_num=0):
    rescale = Rescale(50)
    crop = FaceCrop()
    # transforms.Compose below will apply all of the 
    # transformations sequentially on the input image
    composed = transforms.Compose([Rescale(250),
                                FaceCrop(),
                                Rescale(128),
                                Random90DegFlip()])

    # apply the transforms to a sample image
    sample = dataset[test_num]

    fig = plt.figure()
    for i, tx in enumerate([rescale, crop, composed]):
        transformed_sample = tx(sample)
        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tx).__name__)
        show_keypoints(transformed_sample['image'], transformed_sample['keypoints'])
    plt.show()