import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, weight):
        super(Net, self).__init__()
        k_height, k_width = weight.shape[2:]
        self.conv = nn.Conv2d(in_channels=1, # for gray image
                              out_channels=weight.shape[0],
                              kernel_size=(k_height,
                                           k_width),
                              bias=False)
        self.conv.weight = nn.Parameter(weight)
        self.pool = nn.MaxPool2d(kernel_size=4,
                                 stride=4)

    def forward(self, x):
        conv_output = self.conv(x)
        activatio_output = F.relu(conv_output)
        pooled_output = self.pool(activatio_output)

        return conv_output, activatio_output, pooled_output
    
    def viz_layer(self, layer, tag: str, n_filters= 4):
        fig = plt.figure(figsize=(20, 20))
        print(tag)
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


class ConvNet1(nn.Module):
    def __init__(self):
        super(ConvNet1, self).__init__()
        
        # one 28*28 input image channel (grayscale), 10 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (28-3)/1 +1 = 26
        # the output Tensor for one image, will have the dimensions: (10, 26, 26)
        # after one pool layer, this becomes (10, 13, 13)
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=10,
                               kernel_size=3)
        
        self.pool = nn.MaxPool2d(kernel_size=2,
                                 stride=2)
        
        # second conv layer: 10 inputs, 20 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (13-3)/1 +1 = 11
        # the output tensor will have dimensions: (20, 11, 11)
        # after another pool layer this becomes (20, 5, 5); 5.5 is rounded down
        self.conv2 = nn.Conv2d(in_channels=10,
                               out_channels=20,
                               kernel_size=3)
        
        # 20 outputs * the 5*5 filtered/pooled map size
        # 10 output channels (for the 10 classes)
        self.fc1 = nn.Linear(20*5*5, 10)
        
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        # converting the 10 outputs into a distribution of class scores
        x = F.log_softmax(x, dim=1)
        
        return x
    
    def train_model(self,
              train_loader,
              criterion,
              optimizer,
              n_epochs=30,
              print_every=1000):

        loss_over_time = []
        for epoch in range(n_epochs):
            running_loss = 0.0
            for batch_i, data in enumerate(train_loader):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                if batch_i % int(print_every) == int(print_every)-1:  # print every 1000 batches
                    avg_loss = running_loss/int(print_every)
                    loss_over_time.append(avg_loss)
                    print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, avg_loss))
                    running_loss = 0.0

        print('Finished Training')
        return loss_over_time
    
    def plot_loss(self,
                  loss_vector,
                  print_every=1000):
        
        plt.plot(loss_vector)
        plt.xlabel('{}\'s of batches'.format(int(print_every)))
        plt.ylabel('Loss')
        plt.show()

    def test_model(self,
             test_loader,
             criterion,
             batch_size,
             classes):
        
        test_loss = torch.zeros(1)
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        # set the module to evaluation mode
        self.eval()

        for batch_i, data in enumerate(test_loader):
            inputs, labels = data
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            test_loss = test_loss + ((torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))
            
            # get the predicted class from the maximum value in the output-list of class scores
            _, predicted = torch.max(outputs.data, 1)
            
            # compare predictions to their true labels
            # creates a `correct` Tensor that holds the number of correctly classified images in a batch
            correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))
            
            # calculate test accuracy for *each* object class
            # we get the scalar value of correct items for a class, by calling `correct[i].item()`
            for i in range(batch_size):
                label = labels.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        print('Test Loss: {:.6f}\n'.format(test_loss.numpy()[0]))

        for i in range(10):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
                
        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
        
    def test_visualize(self,
                       net,
                       test_loader,
                       batch_size,
                       classes):
        
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        # get predictions
        preds = np.squeeze(net(images).data.max(1, keepdim=True)[1].numpy())
        images = images.numpy()

        fig = plt.figure(figsize=(25, 4))
        for idx in np.arange(batch_size):
            ax = fig.add_subplot(2, batch_size//2, idx+1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(images[idx]), cmap='gray')
            ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                        color=("green" if preds[idx]==labels[idx] else "red"))
            
    def save_model(self,
                   model_dir='saved_models/',
                   model_name='fashion_net_simple.pt'):
        
        torch.save(self.state_dict(), model_dir+model_name)