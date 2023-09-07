import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from net_utils import PointNet
from data_utils import train_transforms,PointCloudData
import device
from test_utils import plot_confusion_matrix

os.system("clear")

root_dir = "./10_3D-Images/PointNet/data/ModelNet10/"
test_ds = PointCloudData(root_dir=root_dir,
                          valid=True,
                          folder="test",
                          transform=train_transforms())
classes = test_ds.classes
test_loader = DataLoader(dataset=test_ds, batch_size=64)

pointnet = PointNet().to(device.device)
saved_model_name = "save_0.pth"
pointnet.load_state_dict(torch.load("./10_3D-Images/PointNet/saved_models/" + saved_model_name))

all_labels, all_preds = pointnet.test_model(test_loader=test_loader)
cm = confusion_matrix(all_labels, all_preds)
print(cm)
plot_confusion_matrix(cm, list(classes.keys()), normalize=True)