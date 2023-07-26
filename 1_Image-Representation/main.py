# %%

import cv2
import numpy as np
import matplotlib.image as mpimg # For reading images
import matplotlib.pyplot as plt

# %%

img = mpimg.imread('imgs/car.jpg')
print("Image dimensions: {}".format(img.shape))
img_gray = cv2.cvtColor(img,
                        cv2.COLOR_RGB2GRAY)
plt.imshow(img_gray, cmap='gray')

# %%

x, y = 200, 100
print("Pixcel intensity: {}".format(img_gray[y, x]))

# %%

r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

figure, plots = plt.subplots(ncols=3, nrows=1)
for i, subplot in zip(range(3), plots):
    temp = np.zeros(img.shape, dtype='uint8')
    temp[:,:,i] = img[:,:,i]
    subplot.imshow(temp)
    subplot.set_axis_off()
plt.show()

# %%   ***** PIZZA IN BLUE SCREEN MERGING *****

# np.copy enables us to assign values to the tensor elements
bluescreen = np.copy(mpimg.imread('imgs/bluescreen.jpg'))
background = np.copy(mpimg.imread('imgs/background.jpg')
                     )[0:514, 0:816]

# removing the blue background from the pizza image
bluescreen[bluescreen[:,:,2]>240] = 0

# masking pizza section in the background image
background[np.invert(bluescreen[:,:,2] == 0)] = 0

# merging images
result = bluescreen + background

plt.imshow(result)


# %%   ***** BALLOONS RGB VE. HSA *****
balloons = mpimg.imread('imgs/balloons.jpg')
plt.imshow(balloons)

balloons_rgb = np.copy(balloons)
figure, plots = plt.subplots(ncols=3, nrows=1)
for i, subplot in zip(range(3), plots):
    subplot.imshow(balloons_rgb[:,:,i], cmap='gray')
plots[1].set_title('High variation for the value of different RGB channels')
plt.show()

balloons_hsv = cv2.cvtColor(np.copy(balloons),
                            cv2.COLOR_RGB2HSV)

figure, plots = plt.subplots(ncols=3, nrows=1)
for i, subplot in zip(range(3), plots):
    subplot.imshow(balloons_hsv[:,:,i], cmap='gray')
plots[1].set_title('Hue level (first channel) is pretty consistent')
plt.show()

# %%   ***** PINK BALLOON SELECTION *****

# defining pink threshold in RGB is inaccurate and challenging!
rgb_lower = np.array([180,0,100])
rgb_upper = np.array([255,255,230])

# defining pink threshold in HSV is more accurate and easy!
hsv_lower = np.array([160,0,0])
hsv_upper = np.array([180,255,255])


rgb_mask = cv2.inRange(balloons_rgb,
                       rgb_lower,
                       rgb_upper)
balloons_rgb[rgb_mask==0] = 0

hsv_mask = cv2.inRange(balloons_hsv,
                       hsv_lower,
                       hsv_upper)
balloons_copy = np.copy(balloons)
balloons_copy[hsv_mask==0] = 0

figure, (ax1, ax2) = plt.subplots(ncols=2, nrows=1)
ax1.imshow(balloons_rgb)
ax1.set_title('RGB selection')
ax2.imshow(balloons_copy)
ax2.set_title('HSV selection')

# %%   ***** CAR IN GREEN SCREEN WITH(OUT) SHADOW RGB SELECTION *****

car = mpimg.imread('imgs/car_green_screen.jpg')
car_shadow = mpimg.imread('imgs/car_green_screen_shadow.jpg')

figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.imshow(car)
ax2.imshow(car_shadow)

car_cropped = np.copy(car)
car_cropped_shadow = np.copy(car_shadow)

threshold = 200
car_cropped[car_cropped[:,:,1]>threshold] = 0
car_cropped_shadow[car_cropped_shadow[:,:,1]>threshold] = 0

figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.imshow(car_cropped)
ax2.imshow(car_cropped_shadow)

# %%   ***** CAR IN GREEN SCREEN WITH(OUT) SHADOW HSV SELECTION *****

car_copy = np.copy(car)
car_shadow_copy = np.copy(car_shadow)
car_hsv = np.copy(cv2.cvtColor(car_copy,
                               cv2.COLOR_RGB2HSV))
car_shadow_hsv = np.copy(cv2.cvtColor(car_shadow_copy,
                                      cv2.COLOR_RGB2HSV))

hsv_lower = np.array([50,0,0])
hsv_upper = np.array([65,255,255])

hsv_mask = cv2.inRange(car_hsv,
                       hsv_lower,
                       hsv_upper)
car_copy[hsv_mask!=0] = 0

hsv_mask = cv2.inRange(car_shadow_hsv,
                       hsv_lower,
                       hsv_upper)
car_shadow_copy[hsv_mask!=0] = 0

figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.imshow(car_copy)
ax2.imshow(car_shadow_copy)

# %%


