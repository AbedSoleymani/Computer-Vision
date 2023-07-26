# %%

import cv2
import numpy as np
from functions import fft
import matplotlib.pyplot as plt

# %% FAST FOURIER TRANSFORM (FFT) OF AN IMAGE

image_stripes = cv2.imread('imgs/stripes.jpg')
# cv2.imread returns a BGR image
image_stripes = cv2.cvtColor(image_stripes,
                            cv2.COLOR_BGR2RGB)
gray_stripes = cv2.cvtColor(image_stripes,
                            cv2.COLOR_RGB2GRAY)
norm_gray_stripes = gray_stripes / 255.0

image_solid = cv2.imread('imgs/solid_pink.jpg')
image_solid = cv2.cvtColor(image_solid,
                           cv2.COLOR_BGR2RGB)
solid_gray = cv2.cvtColor(image_solid,
                           cv2.COLOR_RGB2GRAY)
norm_solid_gray = solid_gray / 255.0

f_stripes = fft(norm_gray_stripes)
f_solid = fft(norm_solid_gray)

figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
ax1.imshow(image_stripes)
ax1.set_title('stripes image')
ax2.imshow(image_solid)
ax2.set_title('solid pink image')
ax3.imshow(f_stripes, cmap='gray')
ax3.set_title('FFT of stripes image')
ax4.imshow(f_solid, cmap='gray')
ax4.set_title('FFT of solid pink image')
# %% EDGE DECTION FILTERS: SOBEL

brain = cv2.imread('imgs/brain.jpg')
brain = cv2.cvtColor(brain, cv2.COLOR_BGR2RGB)
brain = cv2.cvtColor(brain, cv2.COLOR_RGB2BGR)
brain = cv2.cvtColor(brain, cv2.COLOR_RGB2GRAY)

# creating custom kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
sobel_y = sobel_x.transpose()

""" when ddepth is set to -1, it means that the
    output image will have the same depth as the input image
"""
x_filter_brain = cv2.filter2D(src=brain,
                              ddepth=-1,
                              kernel=sobel_x)
y_filter_brain = cv2.filter2D(src=brain,
                              ddepth=-1,
                              kernel=sobel_y)

figure, (ax1, ax2, ax3) = plt.subplots(nrows=1,
                                       ncols=3,
                                       figsize=(20,20))
ax1.imshow(brain, cmap='gray')
ax2.imshow(x_filter_brain, cmap='gray')
ax3.imshow(y_filter_brain, cmap='gray')

# %% GAUSSIAB BLURE

blured_brain = cv2.GaussianBlur(src=brain,
                                ksize=(9,9),
                                sigmaX=0,
                                sigmaY=0)

x_filter_blured_brain = cv2.filter2D(src=blured_brain,
                                     ddepth=-1,
                                     kernel=sobel_x)
y_filter_blured_brain = cv2.filter2D(src=blured_brain,
                                     ddepth=-1,
                                     kernel=sobel_y)

figure, (ax1, ax2, ax3) = plt.subplots(nrows=1,
                                       ncols=3,
                                       figsize=(20,20))
ax1.imshow(blured_brain, cmap='gray')
ax2.imshow(x_filter_blured_brain, cmap='gray')
ax3.imshow(y_filter_blured_brain, cmap='gray')

# %% 

phone = cv2.imread('imgs/phone.jpg')
phone = cv2.cvtColor(phone, cv2.COLOR_BGR2RGB)
gray_phone = cv2.cvtColor(phone, cv2.COLOR_RGB2GRAY)

low_threshold = 50
high_threshold = 100
edges_phone = cv2.Canny(gray_phone,
                        low_threshold,
                        high_threshold)

rho = 1
theta = np.pi/180
threshold = 60
min_line_length = 50
max_line_gap = 5
lines = cv2.HoughLinesP(edges_phone,
                        rho, theta,
                        threshold,
                        np.array([]),
                        min_line_length,
                        max_line_gap)

line_phone = np.copy(phone)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_phone,(x1,y1),(x2,y2),(255,0,0),5)

figure, (ax1, ax2, ax3) = plt.subplots(nrows=1,
                                       ncols=3,
                                       figsize=(20,20))
ax1.imshow(phone)
ax2.imshow(edges_phone, cmap='gray')
ax3.imshow(line_phone)

# %%
