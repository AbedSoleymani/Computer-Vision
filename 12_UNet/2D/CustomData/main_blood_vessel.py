import os
import torch
import torch.nn as nn
import torch.optim as optim

from model import UNet
from transforms import transforms
from train import train_fn
from utils import get_loaders, load_checkpoint, save_checkpoint, check_accuracy, save_predictions_as_imgs

# device = "cuda" if torch.cuda.is_available() else "cpu" # for Google Colab
device = "mps" if torch.backends.mps.is_available() else "cpu" # for Apple Silicon
print(device)

# Creating directories for saving model's checkpoints and also performance in terms of images
os.makedirs("./12_UNet/2D/CustomData/checkpoints/", exist_ok=True)
os.makedirs("./12_UNet/2D/CustomData/saved_images/", exist_ok=True)

learning_rate = 1e-4
batch_size = 5
num_epochs = 20
image_height = 512
image_width = image_height
load_model = True
train_img_dir = "./12_UNet/2D/CustomData/RetinalBloodVessels/train_images/"
train_mask_dir = "./12_UNet/2D/CustomData/RetinalBloodVessels/train_masks/"
val_img_dir = "./12_UNet/2D/CustomData/RetinalBloodVessels/val_images/"
val_mask_dir = "./12_UNet/2D/CustomData/RetinalBloodVessels/val_masks/"

model = UNet(in_channels=3, out_channels=1, residual=True).to(device)

"""
`BCEWithLogitsLoss` applies Sigmoid activation over the final layer and calculates the nn.BCELoss.
It is often preferred over applying a Sigmoid activation in the model and then using nn.BCELoss.
This preference is due to the numerical stability and efficiency of the training procedure.
"""
loss_fn = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_transform, val_transform = transforms(image_height=image_height,
                                            image_width=image_width)

train_loader, val_loader = get_loaders(train_dir=train_img_dir,
                                    train_maskdir=train_mask_dir,
                                    val_dir=val_img_dir,
                                    val_maskdir=val_mask_dir,
                                    batch_size=batch_size,
                                    train_transform=train_transform,
                                    val_transform=val_transform)

if load_model:
    load_checkpoint(torch.load("./12_UNet/2D/CustomData/checkpoints/retina_checkpoint.pth.tar"), model)

dice_buffer = check_accuracy(val_loader, model, device=device)

for epoch in range(num_epochs):
    print("epoch: ", epoch+1)
    train_fn(train_loader, model, optimizer, loss_fn, device)

    dice_score = check_accuracy(val_loader, model, device=device)

    if dice_score > dice_buffer:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename="./12_UNet/2D/CustomData/checkpoints/retina_checkpoint.pth.tar")
        dice_buffer = dice_score

    save_predictions_as_imgs(val_loader, model, folder="./12_UNet/2D/CustomData/Retina_saved_images/", device=device)