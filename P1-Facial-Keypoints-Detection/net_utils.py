import torch
import torch.nn as nn
import numpy as np
import csv
import matplotlib.pyplot as plt

from gen_tr_val_ts_datasets import create_datasets

def init_weights(layer):
    for part in layer:
        if type(part) == nn.Linear:
            torch.nn.init.xavier_normal_(part.weight, gain=nn.init.calculate_gain('relu'))
            part.bias.data.fill_(0.01)


# Test the net ona all test images and return average loss

def validation_loss(net, valid_loader, criterion, device):
    net.eval()
    loss = 0.0
    running_loss = 0.0

    for i, batch in enumerate(valid_loader):
        images = batch['image']
        key_pts = batch['keypoints']
        
        # flatten pts
        key_pts = key_pts.view(key_pts.size(0), -1)
                
        # convert variables to floats for regression loss
        key_pts = key_pts.type(torch.FloatTensor).to(device)
        images = images.type(torch.FloatTensor).to(device)

        output_pts = net(images)
        
        loss = criterion(output_pts, key_pts)
        running_loss += loss.item()
    avg_loss = running_loss/(i+1)
    net.train()
    return avg_loss

class EarlyStopping:
    """Early stops the training if validation loss dosen't improve after a given patience."""
    def __init__(self,patience=15):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. 
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
    
    def __call__(self, val_loss, model):
        
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), './P1-Facial-Keypoints-Detection/saved_models/checkpoint.pt')
        self.val_loss_min = val_loss


def train_net(net, n_epochs, img_size, batch_size, scheduler, criterion, optimizer, device):
    
    train_loader, test_loader, valid_loader = create_datasets(batch_size,img_size)
    
    train_loss_over_time = [] # to track the loss as the network trains
    val_loss_over_time = [] # to track the validation loss as the network trains
    
    early_stopping = EarlyStopping()
    
    
    # prepare the net for training
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_train_loss = 0.0
        avg_val_loss = 0.0
        avg_train_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor).to(device)
            images = images.type(torch.FloatTensor).to(device)

            # forward pass to get outputs
            output_pts = net(images)
            
            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
            running_train_loss += loss.item()
            if batch_i % 10 == 9 or batch_i == 0:    # print every 10 batches
                if batch_i == 0:
                    avg_train_loss = running_train_loss
                else:
                    avg_train_loss = running_train_loss/10
                avg_val_loss = validation_loss(net, valid_loader, criterion, device)
                train_loss_over_time.append(avg_train_loss)
                val_loss_over_time.append(avg_val_loss)
                print(f'Epoch: {epoch + 1}, Batch: {batch_i+1}/{3462//batch_size}, Avg. Training Loss: {avg_train_loss:.5f}, Avg. Validation Loss: {avg_val_loss:.5f}')
                running_train_loss = 0.0
        
        # reduce learning rate when avg_val_loss has stopped improving
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, net)
        
        if early_stopping.early_stop:
            # remove data collected after last checkpoint
            train_loss_over_time = train_loss_over_time[:-early_stopping.patience]
            val_loss_over_time = val_loss_over_time[:-early_stopping.patience]
            print("Early stopping")
            break
    
    print('Finished Training')           
    return  train_loss_over_time, val_loss_over_time, epoch + 1


def write_list_to_file(list_values, filename):
    """Write the list to csv file."""
    with open(filename, "w") as outfile:
        for entries in list_values:
            outfile.write(str(entries))
            outfile.write("\n")

def net_sample_output(net, data_loader):
    net.eval()

    for i, sample in enumerate(data_loader):

        images = sample['image']
        key_pts = sample['keypoints']

        images = images.type(torch.FloatTensor)

        output_pts = net(images)
        
        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        
        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts

def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Shows image with predicted keypoints"""
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')

def visualize_output(images, outputs, gt_pts, title: str, num_imgs=10):
    plt.figure(figsize=(20,10))
    for i in range(num_imgs):
        ax = plt.subplot(2, num_imgs//2, i+1)

        image = images[i].data   # get the image from it's wrapper
        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

        predicted_key_pts = outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        '''undo normalization of keypoints''' 
        predicted_key_pts = predicted_key_pts*(image.shape[0]/4)+image.shape[0]/2
        
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]         
            ground_truth_pts = ground_truth_pts*(image.shape[0]/4)+image.shape[0]/2
        
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
            
        plt.axis('off')
    plt.suptitle(title)
    plt.show()