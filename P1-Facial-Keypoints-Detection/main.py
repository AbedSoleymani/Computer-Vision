import os
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler

from generate_dataset import generate_dataset
from net_utils import train_net
from models import NaimishNet
from gen_tr_val_ts_datasets import create_datasets

os.system("clear")
generate_dataset()

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

batch_size = 16
img_size = 224
n_epochs = 2

train_loader, valid_loader, test_loader = create_datasets(batch_size=batch_size,
                                                          img_size=img_size)
net = NaimishNet(img_size, use_maxp=False).to(device)

criterion = nn.MSELoss() # Since it is actually a regression problem
optimizer = optim.Adam(params = net.parameters())
plateau_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',  patience=7, verbose=True)

train_loss, val_loss, epochs = train_net(net=net,
                                        n_epochs=n_epochs,
                                        img_size=img_size,
                                        batch_size=batch_size,
                                        scheduler=plateau_lr_scheduler,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        device=device)


model_dir = './P1-Facial-Keypoints-Detection/saved_models/'
model_name = 'abed_model_epochs'+str(n_epochs)

torch.save(net.state_dict(), model_dir+model_name)
