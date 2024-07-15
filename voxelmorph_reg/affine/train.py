import os
import numpy as np
import torch
import sys
dataloader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(dataloader_path)
import torch.nn as nn
import glob
from dataloader import *
from network import AffineCNN, loss_net

import matplotlib.pyplot as plt
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


training_imgs = glob.glob("/home/hkhz/remote_mnt/data/Training/target/*.jpg")
valid_imgs = glob.glob("/home/hkhz/remote_mnt/data/Validation/target/*.jpg")
tc,vc = init_parameters()
tc.is_mat = False
vc.is_mat = False


train_dataset_iter = TransformImageBatchLoader(training_imgs, tc, tc.num_slices, is_testing=False,
                                               n_parallel_calls=tc.n_threads, q_limit=tc.q_limit,
                                               n_epoch=2, gray = True)
valid_dataset_iter = TransformImageBatchLoader(valid_imgs, vc, vc.num_slices, is_testing=True,
                                              n_parallel_calls=vc.n_threads, q_limit=vc.q_limit,
                                              n_epoch=2, gray=True)

n_epochs = 500

model = AffineCNN((256,256)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
# TRAINING
train_loss = []
train_l2_loss = []
train_ncc_loss = []
for _ in range(n_epochs):
    train_iter = iter(train_dataset_iter)
    #valid_iter = iter(valid_dataset_iter)
    data = next(train_iter)
    src = data['target'].to(device)
    print(src.shape)
    tgt = data['input'].to(device)
    optimizer.zero_grad()
    model.train()
    
    with torch.no_grad():
        moved_img, aff = model(src, tgt)
    
        
    loss, l2, ncc = loss_net(moved_img, tgt)
    train_loss.append(loss)
    train_l2_loss.append(l2)
    train_ncc_loss.append(ncc)
    
    loss.backward()
    
    optimizer.step()
    
    print(f"Total loss for epoch {_}: ", loss.item())
    print(f"L2 loss for epoch {_}: ", l2.item())
    print(f"NCC loss for epoch {_}: ", ncc.item())
        
        
#EVALUATION
eval_steps = 50
model.eval()
loss_eval = 0
ncc_eval =0
l2_eval = 0
loss_eval_list = []
l2_eval_list = []
ncc_eval_list = []
for _ in range(eval_steps):
    valid_iter = iter(valid_dataset_iter)
    
    data_v = next(valid_iter)
    src = data_v['target'].to(device)
    tgt = data_v['input'].to(device)
    
    with torch.no_grad():
        moved_img, aff = model(src, tgt)
    loss, l2, ncc = loss_net(moved_img, tgt)
    loss_eval_list.append(loss)
    l2_eval_list.append(loss)
    ncc_eval_list.append(loss)
    loss_eval+=loss.item()
    ncc_eval += ncc.item()
    l2_eval += l2.item()
    
print("avg evaluation total loss: ", loss_eval/eval_steps)
print("avg evaluation total loss: ", l2_eval/eval_steps)
print("avg evaluation total loss: ", ncc_eval/eval_steps)

    
    

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(train_l2_loss, label='L2 Loss')
plt.plot(train_ncc_loss, label='NCC Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Plot evaluation loss
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
plt.savefig("./voxelmorph_reg/affine/losses.jpg")