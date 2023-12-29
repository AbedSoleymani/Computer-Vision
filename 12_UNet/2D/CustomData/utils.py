import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="./12_UNet/2D/CustomData/checkpoints/checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_dir, train_maskdir,
                val_dir, val_maskdir,
                batch_size,
                train_transform, val_transform):
    
    train_ds = CarvanaDataset(image_dir=train_dir,
                              mask_dir=train_maskdir,
                              transform=train_transform)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True)

    val_ds = CarvanaDataset(image_dir=val_dir,
                            mask_dir=val_maskdir,
                            transform=val_transform)

    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            shuffle=False)

    return train_loader, val_loader

def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    
    return dice_score/len(loader)

def save_predictions_as_imgs(loader, model, folder, device):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

        model.train()

def plot_attention_map_grid(attention_maps, num_rows, num_columns, x_label=None, y_label=None):
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 3))
    plt.subplots_adjust(wspace=0.05, hspace=-0.21)  # Adjust spacing between subplots

    for i in range(num_rows):
        for j in range(num_columns):
            index = i * num_columns + j
            ax = axs[i, j]

            if len(attention_maps[index].shape) == 2:
                im = ax.imshow(attention_maps[index], cmap='RdBu_r', interpolation='nearest', aspect='auto')
            else:
                im = ax.imshow(attention_maps[index][0], cmap='RdBu_r', interpolation='nearest', aspect='auto')

            # Set axes with equal metrics
            ax.set_aspect('equal')
            ax.axis('off')  # Turn off axis labels and ticks

    # Add a smaller colorbar to the right of the grid
    cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)

    plt.show()