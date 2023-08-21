from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import os

from facial_keypoints_dataset import FacialKeypointsDataset
from preprocessing import Rescale, RandomRotate, RandomHorizontalFlip, RandomCrop, Normalize, ColorJitter, ToTensor, FaceCropTight, FaceCrop

def create_datasets(batch_size,
                    img_size,
                    validation_size=0.5,
                    color=False):

    train_transform = transforms.Compose([RandomRotate(5),
                                          RandomHorizontalFlip(),
                                          FaceCrop(),
                                          Rescale((img_size,img_size)),
                                          Normalize(color=color),
                                          ToTensor()])

    test_transform = transforms.Compose([FaceCropTight(),
                                         Rescale((img_size,img_size)),
                                         Normalize(color=color),
                                         ToTensor()])
                
    os.chdir("./P1-Facial-Keypoints-Detection/")
    train_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                           root_dir='data/training/',
                                           transform=train_transform)
    
    test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
                                          root_dir='data/test/',
                                          transform=test_transform)
            

    num_test = len(test_dataset)
    indices = list(range(num_test))
    np.random.shuffle(indices)
    split = int(np.floor(validation_size * num_test))
    test_idx, valid_idx = indices[split:], indices[:split]
    
    # define samplers for obtaining training and validation batches
    test_sampler = SubsetRandomSampler(test_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # loading each data in batches
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size,
                              shuffle=True, 
                              num_workers=0)
    
    valid_loader = DataLoader(test_dataset,
                              batch_size=batch_size, 
                              sampler=valid_sampler,
                              num_workers=0)
    
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size,
                             sampler=test_sampler, 
                             num_workers=0)
    
    return train_loader, valid_loader, test_loader