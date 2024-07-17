import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import cv2
import matplotlib.pyplot as plt
from network import AffineCNN, loss_net
from aligners.layers import SpatialTransformer
from GAN_torch.losses_for_gan import NCC
from aligners.aligner_affine import AlignerUNetCVPR2018V4
from IQA_pytorch import DISTS
import random
import sys
import tempfile
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch_msssim import SSIM

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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
    array = tensor.cpu().numpy()
    array = np.squeeze(array)
    array = (array * 255).astype(np.uint8)
    cv2.imwrite(path, array)

def train(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    not_stained_dir = '/home/hkhz/remote_mnt44-50/patches/not_stained_patches_split1024/'
    stained_dir = '/home/hkhz/remote_mnt44-50/patches/stained_patches_split1024/'
    dataset = CustomImageDataset(stained_dir, not_stained_dir, 1000)
    dataset_v = CustomImageDataset(stained_dir, not_stained_dir, 200)

    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler, num_workers=4)

    model = AlignerUNetCVPR2018V4(vol_size=(1024, 1024), enc_nf=[16, 32, 64, 128], dec_nf=[128, 64, 32, 16], src_feats=1, tgt_feats=1).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-5, weight_decay=1e-5)
    criterion = CombinedLoss(alpha=0.5)

    train_loss = []
    train_l1_loss = []
    train_ssim_loss = []

    n_epochs = 150
    for epoch in range(n_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for data in train_loader:
            src = data['input'].to(rank)
            tgt = data['target'].to(rank)
            optimizer.zero_grad()

            moved_img, aff = model(src, tgt)
            loss, l1, ssim = criterion(moved_img, tgt)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(train_loader)

        if rank == 0:
            train_loss.append(avg_loss)
            print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {avg_loss}")

            save_dir = './models'
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f'epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, model_path)

    if rank == 0:
        evaluate(ddp_model, dataset_v, rank)

    cleanup()

def evaluate(model, dataset_v, rank):
    eval_save_dir = './evaluation_results_new'
    os.makedirs(eval_save_dir, exist_ok=True)

    eval_steps = 50
    model.eval()
    loss_eval = 0
    loss_eval_list = []
    l1_eval_list = []
    ssim_eval_list = []

    valid_loader = DataLoader(dataset_v, batch_size=2, shuffle=False, num_workers=4)
    criterion = CombinedLoss(alpha=0.5)

    for step, data_v in enumerate(valid_loader):
        if step >= eval_steps:
            break
        src = data_v['input'].to(rank, non_blocking=True)
        tgt = data_v['target'].to(rank, non_blocking=True)

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

    avg_eval_loss = loss_eval / eval_steps
    print(f"Avg Evaluation Loss: {avg_eval_loss}")
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(train_l1_loss, label='L1 Loss')
    plt.plot(train_ssim_loss, label='SSIM Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()"""

    plt.subplot(1, 2, 2)
    plt.plot(loss_eval_list, label='Total Evaluation Loss')
    plt.plot(l1_eval_list, label='L2 Evaluation Loss')
    plt.plot(ssim_eval_list, label='SSIM Evaluation Loss')
    plt.xlabel('Evaluation Steps')
    plt.ylabel('Loss')
    plt.title('Evaluation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig("./losses.jpg")

def run(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    run(train, world_size)
