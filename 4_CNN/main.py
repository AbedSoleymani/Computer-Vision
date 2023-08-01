# %%

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from models import *
from fasionMNIST import Data

# %% READING THE IMAGE

bgr_img = cv2.imread('imgs/car.jpg')
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# normalize, rescale entries to lie in [0,1]
gray_img = gray_img.astype("float32")/255

plt.imshow(gray_img, cmap='gray')
plt.show()

# %% DEFINING A COSTUM FILTER

filter_vals = np.array([[-1, -1, 1, 1],
                        [-1, -1, 1, 1],
                        [-1, -1, 1, 1],
                        [-1, -1, 1, 1]])

filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1,
                    filter_2,
                    filter_3,
                    filter_4])

fig = plt.figure(figsize=(10, 5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))
    width, height = filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(filters[i][x][y]), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if filters[i][x][y]<0 else 'black')

# %%
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight=weight)
print(model)

gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)
conv_output, activatio_output, pooled_output = model.forward(gray_img_tensor)

model.viz_layer(layer=conv_output,
                tag='conv_output:')
model.viz_layer(layer=activatio_output,
                tag='activatio_output:')
model.viz_layer(layer=pooled_output,
                tag='pooled_output:')

# %%
batch_size = 20
fashionMNIST = Data(batch_size=batch_size)
train_loader, test_loader, classes = fashionMNIST.generate()
fashionMNIST.visualize(data_generator=train_loader,
                       classes=classes)

# %%

convnet1 = ConvNet1()
print(convnet1)

# cross entropy loss combines softmax and nn.NLLLoss() in one single class.
criterion = nn.NLLLoss()
optimizer = optim.SGD(convnet1.parameters(), lr=0.001)

training_loss = convnet1.train_model(train_loader=train_loader,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     n_epochs=5)

convnet1.plot_loss(loss_vector=training_loss)

# %%

convnet1.test_model(test_loader=test_loader,
                    criterion=criterion,
                    batch_size=batch_size,
                    classes=classes)

convnet1.test_visualize(net=convnet1,
                        test_loader=test_loader,
                        batch_size=batch_size,
                        classes=classes)

convnet1.save_model()

# %%


