import torch
import os
from torch.utils.data import DataLoader
import numpy as np

from generate_dataset import generate_dataset
from dataset_utils import show_sample, open_object, point_clouds_show
from data_utils import default_transforms, train_transforms,PointCloudData
from point_sampler import PointSampler
from points2stl import points2stl

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
                          transform=default_transforms())
valid_ds = PointCloudData(root_dir=root_dir,
                          valid=True,
                          folder="test",
                          transform=default_transforms())

print('Train dataset size: ', len(train_ds))
print('Valid dataset size: ', len(valid_ds))
print('Number of classes: ', len(train_ds.classes))
print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
print('Class: ', train_ds.classes)

points = train_ds[0]['pointcloud'].cpu().numpy()
np.save('points.npy', points)
points2stl(points=points, file_name='mesh')
# point_clouds_show(*points.T)

# train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
# valid_loader = DataLoader(dataset=valid_ds, batch_size=64)