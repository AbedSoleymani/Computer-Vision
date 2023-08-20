import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from torchvision import transforms

from generate_dataset import generate_dataset
from utils import show_rand_img, show_keypoints, show_preprocessing_results
from facial_keypoints_dataset import FacialKeypointsDataset
from preprocessing import Rescale, RandomCrop, FaceCrop, Normalize, ToTensor, Random90DegFlip



os.system("clear")
generate_dataset()

os.chdir("./P1-Facial-Keypoints-Detection")
face_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                      root_dir='./data/training/',
                                      transform=None)

# show_rand_img(dataset=face_dataset)

# show_preprocessing_results(dataset=face_dataset, test_num=0)

"""Now, we are ready to create a processed dataset suitable for training"""
data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])
transformed_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                             root_dir='./data/training/',
                                             transform=data_transform)

"""Note that in the transformed dataset, facial keypoints
belong into [-2, 2] and they do not match the image"""
show_rand_img(dataset=transformed_dataset)


