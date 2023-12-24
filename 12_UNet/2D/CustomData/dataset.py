import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 transform=None):
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        """Next line of code is for disregarding `.DS_Store` in Mac"""
        self.images = [filename for filename in os.listdir(image_dir) if filename != '.DS_Store']


    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) # converting to gray scale

        mask[mask == 255.0] = 1.0 # making the mask binary for the softmax output of the network

        # The same transformation from the `albumentations` library will be applied on images and masks.
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
