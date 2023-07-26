# %%

import matplotlib.pyplot as plt
import numpy as np
import cv2

# %% CORNER DETECTION

waffle = cv2.imread('imgs/waffle.jpg')
waffle = cv2.cvtColor(waffle, cv2.COLOR_BGR2RGB)
gray_waffle = cv2.cvtColor(waffle, cv2.COLOR_RGB2GRAY)
gray_waffle = np.float32(gray_waffle)

corners = cv2.cornerHarris(gray_waffle, 2, 3, 0.04)
corners = cv2.dilate(corners,None)

thresh = 0.05 * corners.max()
corner_image = np.copy(waffle)

for j in range(0, corners.shape[0]):
    for i in range(0, corners.shape[1]):
        if(corners[j,i] > thresh):
            cv2.circle(corner_image, (i, j), 1, (0,255,0), 1)

plt.imshow(corner_image)

# %% CONTOUR DETECTION

hands = cv2.imread('imgs/hands.jpg')
hands = cv2.cvtColor(hands, cv2.COLOR_BGR2RGB)
gray_hands = cv2.cvtColor(hands, cv2.COLOR_RGB2GRAY)

retval, binary_hands = cv2.threshold(src=gray_hands,
                                     thresh=225,
                                     maxval=255,
                                     type=cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(image=binary_hands,
                                       mode=cv2.RETR_TREE,
                                       method=cv2.CHAIN_APPROX_SIMPLE)

hands = cv2.drawContours(image=hands,
                         contours=contours,
                         contourIdx=-1, # all countors
                         color=(0,255,0),
                         thickness=3)
plt.imshow(hands)

# %% IMAGE SEGMENTATION USING K-MEANS CUSTERING

image = cv2.imread('imgs/oranges.jpg')
# image = cv2.imread('imgs/pancakes.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original = np.copy(image)

# Reshape image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1,3))
# Convert to float type
pixel_vals = np.float32(pixel_vals)

# k-means stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100, # max iteration
            0.2)
k = 3
retval, labels, centers = cv2.kmeans(data=pixel_vals,
                                     K=k,
                                     bestLabels=None,
                                     criteria=criteria,
                                     attempts=10,
                                     flags=cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
labels_reshape = labels.reshape(image.shape[0], image.shape[1])

image, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.imshow(original)
ax1.set_title('original')
ax2.imshow(segmented_image)
ax2.set_title('K = {}'.format(k))

image, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
## Visualizeing one segment
ax1.imshow(labels_reshape==1, cmap='gray')

# mask an image segment by cluster
cluster = 0 
masked_image = np.copy(original)
masked_image[labels_reshape == cluster] = [0, 255, 0]
ax2.imshow(masked_image)

# %% ORB

image = cv2.imread('./imgs/face.jpeg')
training_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
training_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(200, 2.0)
keypoints, descriptor = orb.detectAndCompute(training_gray,
                                             None)
keyp_without_size = np.copy(training_image)
keyp_with_size = np.copy(training_image)

cv2.drawKeypoints(image=training_image,
                  keypoints=keypoints,
                  outImage=keyp_without_size,
                  color = (0, 255, 0))
cv2.drawKeypoints(image=training_image,
                  keypoints=keypoints,
                  outImage=keyp_with_size,
                  flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

image, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.imshow(keyp_without_size)
ax1.set_title('Number of keypoints Detected:  {}'.format(len(keypoints)))
ax2.imshow(keyp_with_size)


# %%
