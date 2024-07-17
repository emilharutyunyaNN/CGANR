"""import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from network import AffineCNN, loss_net
from aligners.layers import SpatialTransformer
from GAN_torch.losses_for_gan import NCC
from aligners.aligner_affine import AlignerUNetCVPR2018V4
from IQA_pytorch import DISTS
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import random
#D = DISTS(channels=1).to(device)

class CustomImageDataset(Dataset):
    def __init__(self, dir1, dir2, num_samples = 1000,transform=None):
        self.stained_dir = dir1
        self.not_stained_dir = dir2
        self.transform = transform
        self.image_pairs = self._get_image_pairs(num_samples)

    def _get_image_pairs(self, num_samples):
        #not_stained_files = os.listdir(self.not_stained_dir)
        stained_files = os.listdir(self.stained_dir)
        #not_stained_dict = {self._extract_key(f): f for f in not_stained_files if f.endswith('.jpg')}
        #stained_dict = {self._extract_key(f): f for f in stained_files if f.endswith('.jpg')}
        #common_numbers = set(not_stained_dict.keys()).intersection(set(stained_dict.keys()))
        image_pairs = [(stained.replace("stained", "not_stained"),stained) for stained in stained_files]
        if num_samples:
            random.shuffle(image_pairs)
            image_pairs = image_pairs[:num_samples]
        return image_pairs

    def _extract_key(self, filename):
        return '_'.join([part for part in filename.split('_') if part.isdigit()])

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        not_stained_name, stained_name = self.image_pairs[idx]
        #print(not_stained_name, stained_name)
        not_stained_path = os.path.join(self.not_stained_dir, not_stained_name)
        stained_path = os.path.join(self.stained_dir, stained_name)
        not_stained_img = cv2.imread(not_stained_path, cv2.IMREAD_GRAYSCALE)
        stained_img = cv2.imread(stained_path, cv2.IMREAD_GRAYSCALE)
        not_stained_img = np.expand_dims((not_stained_img / 255.0).astype(np.float32), axis=0)#(not_stained_img / 255.0).astype(np.float32).transpose(2,0,1)xv
        stained_img = np.expand_dims((stained_img / 255.0).astype(np.float32), axis=0)#(stained_img / 255.0).astype(np.float32).transpose(2,0,1)#
        
        return {'input': stained_img, 'target': not_stained_img}

not_stained_dir = '/home/hkhz/remote_mnt44-50/patches/not_stained_patches_split1024/'
stained_dir = '/home/hkhz/remote_mnt44-50/patches/stained_patches_split1024/'
dataset = CustomImageDataset(stained_dir, not_stained_dir, 1000)
dataset_v = CustomImageDataset(stained_dir, not_stained_dir, 200)


train_dataset_iter = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)  # Reduced batch size for memory

n_epochs = 150
model = AlignerUNetCVPR2018V4(vol_size = (1024,1024), enc_nf = [16,32,64,128], dec_nf = [128,64,32,16], src_feats=1, tgt_feats=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
scaler = GradScaler()
criterion = nn.L1Loss()
# Create directory for saving models
save_dir = './models'
os.makedirs(save_dir, exist_ok=True)
from pytorch_msssim import SSIM

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1)

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = 1 - self.ssim_loss(pred, target)
        return self.alpha * l1 + (1 - self.alpha) * ssim, l1, ssim

# Usage
criterion = CombinedLoss(alpha=0.5)

# TRAINING
train_loss = []
train_l1_loss = []
train_ssim_loss = []

for epoch in range(n_epochs):
    model.train()
    for data in train_dataset_iter:
        src = data['input'].to(device)
        tgt = data['target'].to(device)
        optimizer.zero_grad()
        
        initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}
        
        #with autocast():  # Use mixed precision
        moved_img, aff = model(src, tgt)
        loss, l1, ssim = criterion(moved_img, tgt)
        
        
        #print(torch.dtype(loss))
        #scaler.scale(loss).backward()
        loss.backward()
        #scaler.step(optimizer)
        #scaler.update()
        optimizer.step()

        #for name, param in model.named_parameters():
          #  if not torch.equal(initial_params[name], param):
           #     print(f"Layer {name} weights have been updated.")
          #  else:
           #     print(f"Layer {name} weights have NOT been updated.")

        train_loss.append(loss.item())
        train_l1_loss.append(l1.item())
        train_ssim_loss.append(ssim.item())

        torch.cuda.empty_cache()  # Clear cache to free up memory

    print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {loss.item()}")
          #L2 Loss: {l2.item()}, NCC Loss: {ncc.item()}"

    # Save the model
    model_path = os.path.join(save_dir, f'epoch_{epoch + 1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, model_path)

# EVALUATION
eval_save_dir = './evaluation_results_new'
os.makedirs(eval_save_dir, exist_ok=True)

# Evaluation
eval_steps = 50
model.eval()
loss_eval = 0
loss_eval_list = []
l1_eval_list = []
ssim_eval_list = []


def save_image(tensor, path):
    
    array = tensor.cpu().numpy()
    array = np.squeeze(array)  # Remove single dimensions (e.g., 1xHxW or 1x1xHxW)
    array = (array * 255).astype(np.uint8)  # Convert to uint8 and scale to [0, 255]
    cv2.imwrite(path, array)

valid_dataset_iter = DataLoader(dataset_v, batch_size=2, shuffle=False, num_workers=4)

for step, data_v in enumerate(valid_dataset_iter):
    if step >= eval_steps:
        break
    src = data_v['input'].to(device, non_blocking=True)
    tgt = data_v['target'].to(device, non_blocking=True)

    with torch.no_grad():
        moved_img, aff = model(src, tgt)
        loss, l1, ssim = criterion(moved_img, tgt)
    
    loss_eval_list.append(loss.item())
    l1_eval_list.append(l1.item())
    ssim_eval_list.append(ssim.item())
    loss_eval += loss.item()

    # Save images
    for i in range(src.size(0)):
        save_image(moved_img[i], os.path.join(eval_save_dir, f'moved_img_{step * src.size(0) + i}.png'))
        save_image(tgt[i], os.path.join(eval_save_dir, f'target_img_{step * src.size(0) + i}.png'))

print(f"Avg Evaluation Loss: {loss_eval / eval_steps}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(train_l1_loss, label='L1 Loss')
plt.plot(train_ssim_loss, label='SSIMs Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_eval_list, label='Total Evaluation Loss')
plt.plot(l1_eval_list, label='L2 Evaluation Loss')
plt.plot(ssim_eval_list, label='NCC Evaluation Loss')
plt.xlabel('Evaluation Steps')
plt.ylabel('Loss')
plt.title('Evaluation Loss')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("./losses.jpg")
"""
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from network import AffineCNN, loss_net
from aligners.layers import SpatialTransformer
from GAN_torch.losses_for_gan import NCC
from aligners.aligner_affine import AlignerUNetCVPR2018V4
from IQA_pytorch import DISTS
import random
from pytorch_msssim import SSIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class remains unchanged
class CustomImageDataset(Dataset):
    def __init__(self, dir1, dir2, num_samples=1000, transform=None):
        self.stained_dir = dir1
        self.not_stained_dir = dir2
        self.transform = transform
        self.image_pairs = self._get_image_pairs(num_samples)

    def _get_image_pairs(self, num_samples):
        stained_files = os.listdir(self.stained_dir)
        image_pairs = [(stained.replace("stained", "not_stained"), stained) for stained in stained_files]
        if num_samples:
            random.shuffle(image_pairs)
            image_pairs = image_pairs[:num_samples]
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        not_stained_name, stained_name = self.image_pairs[idx]
        not_stained_path = os.path.join(self.not_stained_dir, not_stained_name)
        stained_path = os.path.join(self.stained_dir, stained_name)
        not_stained_img = cv2.imread(not_stained_path, cv2.IMREAD_GRAYSCALE)
        stained_img = cv2.imread(stained_path, cv2.IMREAD_GRAYSCALE)
        not_stained_img = np.expand_dims((not_stained_img / 255.0).astype(np.float32), axis=0)
        stained_img = np.expand_dims((stained_img / 255.0).astype(np.float32), axis=0)
        return {'input': stained_img, 'target': not_stained_img}

not_stained_dir = '/home/hkhz/remote_mnt44-50/patches/not_stained_patches_split1024/'
stained_dir = '/home/hkhz/remote_mnt44-50/patches/stained_patches_split1024/'
dataset = CustomImageDataset(stained_dir, not_stained_dir, 1000)
dataset_v = CustomImageDataset(stained_dir, not_stained_dir, 200)

train_dataset_iter = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1)

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = 1 - self.ssim_loss(pred, target)
        return self.alpha * l1 + (1 - self.alpha) * ssim, l1, ssim

model = AlignerUNetCVPR2018V4(vol_size=(1024, 1024), enc_nf=[16, 32, 64, 128], dec_nf=[128, 64, 32, 16], src_feats=1, tgt_feats=1)
model = nn.DataParallel(model)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
scaler = GradScaler()
criterion = CombinedLoss(alpha=0.5)

# TRAINING
train_loss = []
train_l1_loss = []
train_ssim_loss = []

for epoch in range(150):
    model.train()
    for data in train_dataset_iter:
        src = data['input'].to(device)
        tgt = data['target'].to(device)
        optimizer.zero_grad()
        
        moved_img, aff = model(src, tgt)
        loss, l1, ssim = criterion(moved_img, tgt)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_l1_loss.append(l1.item())
        train_ssim_loss.append(ssim.item())

    print(f"Epoch {epoch + 1}/150 - Loss: {loss.item()}")

    # Save the model
    model_path = os.path.join('./models', f'epoch_{epoch + 1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, model_path)

# EVALUATION
eval_save_dir = './evaluation_results_new'
os.makedirs(eval_save_dir, exist_ok=True)

eval_steps = 50
model.eval()
loss_eval = 0
loss_eval_list = []
l1_eval_list = []
ssim_eval_list = []

def save_image(tensor, path):
    array = tensor.cpu().numpy()
    array = np.squeeze(array)
    array = (array * 255).astype(np.uint8)
    cv2.imwrite(path, array)

valid_dataset_iter = DataLoader(dataset_v, batch_size=8, shuffle=False, num_workers=4)

for step, data_v in enumerate(valid_dataset_iter):
    if step >= eval_steps:
        break
    src = data_v['input'].to(device, non_blocking=True)
    tgt = data_v['target'].to(device, non_blocking=True)

    with torch.no_grad():
        moved_img, aff = model(src, tgt)
        loss, l1, ssim = criterion(moved_img, tgt)
    
    loss_eval_list.append(loss.item())
    l1_eval_list.append(l1.item())
    ssim_eval_list.append(ssim.item())
    loss_eval += loss.item()

    for i in range(src.size(0)):
        save_image(moved_img[i], os.path.join(eval_save_dir, f'moved_img_{step * src.size(0) + i}.png'))
        save_image(tgt[i], os.path.join(eval_save_dir, f'target_img_{step * src.size(0) + i}.png'))

print(f"Avg Evaluation Loss: {loss_eval / eval_steps}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(train_l1_loss, label='L1 Loss')
plt.plot(train_ssim_loss, label='SSIM Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_eval_list, label='Total Evaluation Loss')
plt.plot(l1_eval_list, label='L1 Evaluation Loss')
plt.plot(ssim_eval_list, label='SSIM Evaluation Loss')
plt.xlabel('Evaluation Steps')
plt.ylabel('Loss')
plt.title('Evaluation Loss')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("./losses.jpg")
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import cv2
import matplotlib.pyplot as plt
from pytorch_msssim import SSIM
from aligners.aligner_affine import AlignerUNetCVPR2018V4
from IQA_pytorch import DISTS
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Custom Dataset class
class CustomImageDataset(Dataset):
    def __init__(self, dir1, dir2, num_samples=1000, transform=None):
        self.stained_dir = dir1
        self.not_stained_dir = dir2
        self.transform = transform
        self.image_pairs = self._get_image_pairs(num_samples)

    def _get_image_pairs(self, num_samples):
        stained_files = os.listdir(self.stained_dir)
        image_pairs = [(stained.replace("stained", "not_stained"), stained) for stained in stained_files]
        if num_samples:
            np.random.shuffle(image_pairs)
            image_pairs = image_pairs[:num_samples]
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        not_stained_name, stained_name = self.image_pairs[idx]
        not_stained_path = os.path.join(self.not_stained_dir, not_stained_name)
        stained_path = os.path.join(self.stained_dir, stained_name)
        not_stained_img = cv2.imread(not_stained_path, cv2.IMREAD_GRAYSCALE)
        stained_img = cv2.imread(stained_path, cv2.IMREAD_GRAYSCALE)
        not_stained_img = np.expand_dims((not_stained_img / 255.0).astype(np.float32), axis=0)
        stained_img = np.expand_dims((stained_img / 255.0).astype(np.float32), axis=0)
        return {'input': stained_img, 'target': not_stained_img}

# Custom loss function
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1)

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = 1 - self.ssim_loss(pred, target)
        return self.alpha * l1 + (1 - self.alpha) * ssim, l1, ssim

def save_image(tensor, path):
    """Save a tensor as an image file."""
    array = tensor.cpu().numpy()
    array = np.squeeze(array)  # Remove single dimensions (e.g., 1xHxW or 1x1xHxW)
    array = (array * 255).astype(np.uint8)  # Convert to uint8 and scale to [0, 255]
    cv2.imwrite(path, array)

def train(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Create data loaders
    not_stained_dir = '/home/hkhz/remote_mnt44-50/patches/not_stained_patches_split1024/'
    stained_dir = '/home/hkhz/remote_mnt44-50/patches/stained_patches_split1024/'
    dataset = CustomImageDataset(stained_dir, not_stained_dir, 1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=4)

    dataset_v = CustomImageDataset(stained_dir, not_stained_dir, 200)
    sampler_v = DistributedSampler(dataset_v, num_replicas=world_size, rank=rank)
    valid_loader = DataLoader(dataset_v, batch_size=2, sampler=sampler_v, num_workers=4)

    # Initialize model, loss, and optimizer
    model = AlignerUNetCVPR2018V4(vol_size=(1024, 1024), enc_nf=[16, 32, 64, 128], dec_nf=[128, 64, 32, 16], src_feats=1, tgt_feats=1).to(rank)
    model = DDP(model, device_ids=[rank])
    criterion = CombinedLoss(alpha=0.5).to(rank)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    n_epochs = 150
    train_loss = []
    train_l1_loss = []
    train_ssim_loss = []
    
    for epoch in range(n_epochs):
        model.train()
        sampler.set_epoch(epoch)
        for batch in train_loader:
            src = batch['input'].to(rank)
            tgt = batch['target'].to(rank)
            optimizer.zero_grad()
            moved_img, aff = model(src, tgt)
            loss, l1, ssim = criterion(moved_img, tgt)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_l1_loss.append(l1.item())
            train_ssim_loss.append(ssim.item())

        if rank == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {loss.item()}")

        # Save the model
        if rank == 0 and (epoch + 1) % 10 == 0:
            save_dir = './models'
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, model_path)

    # Evaluation
    if rank == 0:
        model.eval()
        eval_save_dir = './evaluation_results'
        os.makedirs(eval_save_dir, exist_ok=True)
        eval_steps = 50
        loss_eval = 0
        loss_eval_list = []
        l1_eval_list = []
        ssim_eval_list = []

        for step, data_v in enumerate(valid_loader):
            if step >= eval_steps:
                break
            src = data_v['input'].to(rank)
            tgt = data_v['target'].to(rank)

            with torch.no_grad():
                moved_img, aff = model(src, tgt)
                loss, l1, ssim = criterion(moved_img, tgt)

            loss_eval_list.append(loss.item())
            l1_eval_list.append(l1.item())
            ssim_eval_list.append(ssim.item())
            loss_eval += loss.item()

            # Save images
            for i in range(src.size(0)):
                save_image(moved_img[i], os.path.join(eval_save_dir, f'moved_img_{step * src.size(0) + i}.png'))
                save_image(tgt[i], os.path.join(eval_save_dir, f'target_img_{step * src.size(0) + i}.png'))

        print(f"Avg Evaluation Loss: {loss_eval / eval_steps}")

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Training Loss')
        plt.plot(train_l1_loss, label='L1 Loss')
        plt.plot(train_ssim_loss, label='SSIMs Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(loss_eval_list, label='Total Evaluation Loss')
        plt.plot(l1_eval_list, label='L2 Evaluation Loss')
        plt.plot(ssim_eval_list, label='NCC Evaluation Loss')
        plt.xlabel('Evaluation Steps')
        plt.ylabel('Loss')
        plt.title('Evaluation Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()
        plt.savefig("./losses.jpg")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
