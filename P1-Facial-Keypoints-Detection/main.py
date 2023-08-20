import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from generate_dataset import generate_dataset
from utils import show_rand_img, show_keypoints, show_preprocessing_results
from facial_keypoints_dataset import FacialKeypointsDataset



os.system("clear")
generate_dataset()

os.chdir("./P1-Facial-Keypoints-Detection")
face_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                      root_dir='data/training/')

# show_rand_img(dataset=face_dataset)

show_preprocessing_results(dataset=face_dataset,
                           test_num=0)



