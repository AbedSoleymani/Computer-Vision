import torch
import os
from torch.utils.data import DataLoader
import numpy as np

from generate_dataset import generate_dataset
from dataset_utils import show_sample, open_object, point_clouds_show
from data_utils import default_transforms, train_transforms,PointCloudData
from point_sampler import PointSampler
from points2stl import points2stl
import device
from net_utils import PointNet

os.system("clear")
generate_dataset()

root = "./10_3D-Images/PointNet/data/ModelNet10/"
object_address = root + "bed/train/bed_0017.off"
verts, faces = open_object(object_address)
# show_sample(verts, faces, type="point_clouds")

root_dir = "./10_3D-Images/PointNet/data/ModelNet10/"
train_ds = PointCloudData(root_dir=root_dir,
                          valid=False,
                          folder="train",
                          transform=train_transforms())
valid_ds = PointCloudData(root_dir=root_dir,
                          valid=True,
                          folder="test",
                          transform=train_transforms())

print('Train dataset size: ', len(train_ds),
      'Valid dataset size: ', len(valid_ds),
      'Number of classes: ', len(train_ds.classes))
print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size(),
      'Classes: ', train_ds.classes)

points = train_ds[0]['pointcloud'].cpu().numpy()
np.save('points.npy', points)
# point_clouds_show(*points.T)

train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=64)

"""
Creating, training, and testing the 3D classifier model
"""

pointnet = PointNet().to(device.device)
# print(pointnet)
optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)

pointnet.train_model(optimizer=optimizer,
                     train_loader=train_loader,
                     val_loader=valid_loader,
                     epochs=1,
                     save=True)

