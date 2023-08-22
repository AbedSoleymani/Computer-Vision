from models import NaimishNet
import torch
from gen_tr_val_ts_datasets import create_datasets
from net_utils import *


batch_size = 16
img_size = 224

train_loader, valid_loader, test_loader = create_datasets(batch_size=batch_size,
                                                          img_size=img_size)
net = NaimishNet(img_size, use_maxp=False)


test_images, test_outputs, gt_pts = net_sample_output(net=net,
                                                      data_loader=train_loader)
visualize_output(images=test_images,
                 outputs=test_outputs,
                 gt_pts=gt_pts,
                 title="Before training")


n_epochs = 2
model_dir = './P1-Facial-Keypoints-Detection/saved_models/'
model_name = 'abed_model_epochs' + str(n_epochs)

net.load_state_dict(torch.load(model_dir + model_name))
test_images, test_outputs, gt_pts = net_sample_output(net=net,
                                                      data_loader=train_loader)
visualize_output(images=test_images,
                 outputs=test_outputs,
                 gt_pts=gt_pts,
                 title="After {} epoch(s) of training".format(n_epochs))