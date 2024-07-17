import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset #, DataLoader
import cv2
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from network import AffineCNN, loss_net
from aligners.layers import SpatialTransformer
from GAN_torch.losses_for_gan import NCC
#from IQA_pytorch import DISTS
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#D = DISTS(channels=1).to(device)
from monai.utils import set_determinism, first
from monai.transforms import (EnsureChannelFirstD,
                             Compose,
                             LoadImageD,
                             RandRotateD,
                            RandZoomD,
                            ScaleIntensityRanged,
                            ToTensorD
                             )
from monai.networks.nets import GlobalNet
from monai.networks.blocks import Warp
from torch.nn import L1Loss
from monai.data import DataLoader, CacheDataset

class CustomImageDataset(Dataset):
    def __init__(self, dir1, dir2, transform=None):
        self.stained_dir = dir1
        self.not_stained_dir = dir2
        self.transform = transform
        self.image_pairs = self._get_image_pairs()

    def _get_image_pairs(self):
        not_stained_files = os.listdir(self.not_stained_dir)
        stained_files = os.listdir(self.stained_dir)
        not_stained_dict = {self._extract_key(f): f for f in not_stained_files if f.endswith('.jpg')}
        stained_dict = {self._extract_key(f): f for f in stained_files if f.endswith('.jpg')}
        common_numbers = set(not_stained_dict.keys()).intersection(set(stained_dict.keys()))
        image_pairs = [(not_stained_dict[num], stained_dict[num]) for num in common_numbers]
        return image_pairs

    def _extract_key(self, filename):
        return '_'.join([part for part in filename.split('_') if part.isdigit()])

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        not_stained_name, stained_name = self.image_pairs[idx]
        not_stained_path = os.path.join(self.not_stained_dir, not_stained_name)
        stained_path = os.path.join(self.stained_dir, stained_name)
        #not_stained_img = cv2.imread(not_stained_path, cv2.IMREAD_GRAYSCALE)
        #stained_img = cv2.imread(stained_path, cv2.IMREAD_GRAYSCALE)
        #not_stained_img = np.expand_dims((not_stained_img / 127 - 1).astype(np.float32), axis=0)
        #stained_img = np.expand_dims((stained_img / 127 - 1).astype(np.float32), axis=0)
        return {'input': stained_path, 'target': not_stained_path}

not_stained_dir = '/home/hkhz/remote_mnt44-50/patches/gray_not_stained_patches/'
stained_dir = '/home/hkhz/remote_mnt44-50/patches/gray_stained_patches/'
dataset = CustomImageDataset(not_stained_dir, stained_dir)

train_dataset_iter = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)  # Reduced batch size for memory
training_data_dict = [
    {"fixed": data['input'], "moving":data['target']} for data in dataset

]
import numpy as np
train_transform = Compose(
    [
        LoadImageD(keys =["fixed", "moving"]),
        EnsureChannelFirstD(keys =["fixed", "moving"]),
        ScaleIntensityRanged(keys =["fixed", "moving"], a_min = 0.0, a_max=255,b_min=0.0, b_max=1.0, clip=True),
        RandRotateD(keys=["moving"], range_x = np.pi/4, prob = 1.0, keep_size=True, mode = "bicubic"),
        RandZoomD(keys=["moving"], min_zoom=0.9, max_zoom=1.1, prob = 1.0, mode = "bicubic",align_corners=False),
        ToTensorD(keys =["fixed", "moving"])
    ]
)
train_ds = CacheDataset(data = training_data_dict[:1000], transform=train_transform, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size = 4, shuffle = True, num_workers=2)

model = GlobalNet(
    image_size=(2048,2048),
    spatial_dims=2,
    in_channels=2,  # moving and fixed
    num_channel_initial=16,
    depth=5
).to(device)

image_loss = L1Loss()
warp_layer = Warp("bicubic", "border").to(device)
optimizer = torch.optim.Adam(model.parameters(), 1e-5)



epoch_num = 200
epoch_loss_values = []

for epoch in range(epoch_num):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{epoch_num}")
    model.train()
    epoch_loss, step = 0, 0
    for batch_data in train_loader:
        step += 1
        optimizer.zero_grad()

        moving = batch_data["moving"].to(device)
        fixed = batch_data["fixed"].to(device)
        ddf = model(torch.cat((moving, fixed), dim=1))
        pred_image = warp_layer(moving, ddf)

        loss = image_loss(pred_image, fixed)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # print(f"{step}/{len(train_ds) // train_loader.batch_size}, "
        #       f"train_loss: {loss.item():.4f}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")