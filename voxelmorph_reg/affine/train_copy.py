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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
        not_stained_img = cv2.imread(not_stained_path, cv2.IMREAD_GRAYSCALE)
        stained_img = cv2.imread(stained_path, cv2.IMREAD_GRAYSCALE)
        not_stained_img = np.expand_dims((not_stained_img / 127 - 1).astype(np.float32), axis=0)
        stained_img = np.expand_dims((stained_img / 127 - 1).astype(np.float32), axis=0)
        return {'input': stained_img, 'target': not_stained_img}

not_stained_dir = '/home/hkhz/remote_mnt44-50/patches/not_stained_patches/'
stained_dir = '/home/hkhz/remote_mnt44-50/patches/stained_patches/'
dataset = CustomImageDataset(not_stained_dir, stained_dir)

train_dataset_iter = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)  # Reduced batch size for memory

n_epochs = 50
model = AffineCNN((2048, 2048)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler()

# Create directory for saving models
save_dir = './models'
os.makedirs(save_dir, exist_ok=True)

# TRAINING
train_loss = []
train_l2_loss = []
train_ncc_loss = []

for epoch in range(n_epochs):
    model.train()
    for data in train_dataset_iter:
        src = data['target'].to(device)
        tgt = data['input'].to(device)
        optimizer.zero_grad()

        with autocast():  # Use mixed precision
            moved_img, aff = model(src, tgt)
            loss, l2, ncc = loss_net(moved_img, tgt)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss.append(loss.item())
        train_l2_loss.append(l2.item())
        train_ncc_loss.append(ncc.item())

        torch.cuda.empty_cache()  # Clear cache to free up memory

    print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {loss.item()}, L2 Loss: {l2.item()}, NCC Loss: {ncc.item()}")

    # Save the model
    model_path = os.path.join(save_dir, f'epoch_{epoch + 1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, model_path)

# EVALUATION
eval_steps = 50
model.eval()
loss_eval = 0
ncc_eval = 0
l2_eval = 0
loss_eval_list = []
l2_eval_list = []
ncc_eval_list = []

valid_dataset_iter = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)  # Assuming same dataset for validation

for _ in range(eval_steps):
    valid_iter = iter(valid_dataset_iter)
    data_v = next(valid_iter)
    src = data_v['target'].to(device)
    tgt = data_v['input'].to(device)

    with torch.no_grad():
        moved_img, aff = model(src, tgt)
    loss, l2, ncc = loss_net(moved_img, tgt)
    loss_eval_list.append(loss.item())
    l2_eval_list.append(l2.item())
    ncc_eval_list.append(ncc.item())
    loss_eval += loss.item()
    ncc_eval += ncc.item()
    l2_eval += l2.item()

print(f"Avg Evaluation Loss: {loss_eval / eval_steps}")
print(f"Avg Evaluation L2 Loss: {l2_eval / eval_steps}")
print(f"Avg Evaluation NCC Loss: {ncc_eval / eval_steps}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(train_l2_loss, label='L2 Loss')
plt.plot(train_ncc_loss, label='NCC Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_eval_list, label='Total Evaluation Loss')
plt.plot(l2_eval_list, label='L2 Evaluation Loss')
plt.plot(ncc_eval_list, label='NCC Evaluation Loss')
plt.xlabel('Evaluation Steps')
plt.ylabel('Loss')
plt.title('Evaluation Loss')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("./affine/losses.jpg")
