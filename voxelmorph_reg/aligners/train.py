from stn_affine import affine_transform
from aligner_affine import AlignerUNetCVPR2018V4
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm

class AlignerData(Dataset):
    def __init__(self, datadir_src, datadir_tgt, ind = None, transform = True):
        self.src_dir = datadir_src
        self.tgt_dir = datadir_tgt
        self.images_src = os.listdir(datadir_src)
        self.images_tgt = os.listdir(datadir_tgt)
        self.transform = transform
        if ind:
            dir_tgt = os.listdir(datadir_tgt)
            self.images_tgt = [dir_tgt[i] for i in ind]
            
    def __getitem__(self, idx):
        
        path_src = os.path.join(self.src_dir, 'not_'+self.images_tgt[idx])
        path_tgt = os.path.join(self.tgt_dir, self.images_tgt[idx])
        
        #print(path_src, path_tgt)
        
        img_src = cv2.imread(path_src, cv2.IMREAD_GRAYSCALE)
        img_tgt = cv2.imread(path_tgt, cv2.IMREAD_GRAYSCALE)
        
        img_src = img_src/255.0
        img_tgt = img_tgt/255.0
        
        #print(img_src, img_tgt)
        if self.transform:
            transform = transforms.Compose([transforms.ToTensor()])
            img_src = transform(img_src)
            img_tgt = transform(img_tgt)
        
        return (img_src, img_tgt)
    def __len__(self):
        return len(self.images_tgt)
    
    
dataset = AlignerData("/home/hkhz/remote_mnt36-45/patches/filtered_not_stained_split","/home/hkhz/remote_mnt36-45/patches/filtered_not_stained_split")
train_size = (int)(0.3*len(dataset))
test_size = (int)(0.1*len(dataset))
train_indces = [i for i in range(train_size)]
test_indces = [i for i in range(train_size, train_size+ test_size)]
train_dataset = AlignerData("/home/hkhz/remote_mnt36-45/patches/filtered_not_stained_split","/home/hkhz/remote_mnt36-45/patches/filtered_stained_split", train_indces)
test_dataset = AlignerData("/home/hkhz/remote_mnt36-45/patches/filtered_not_stained_split","/home/hkhz/remote_mnt36-45/patches/filtered_stained_split", test_indces)

traindata = DataLoader(train_dataset, batch_size= 16, shuffle = True)
test_data = DataLoader(test_dataset, batch_size=16, shuffle = False)

model = AlignerUNetCVPR2018V4(
    vol_size=(256,256),
    enc_nf = [64, 128, 256, 512],
    dec_nf = [512, 256, 128, 64]
).to(device)
criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_values = []
num_epochs = 100
loss_values = []
import time
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    tqdm_traindata = tqdm(traindata, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
    for i, data in enumerate(tqdm_traindata):
        start_data = time.time()
        moving = data[0].float().to(device)
        fixed = data[1].float().to(device)
        optimizer.zero_grad()
        transformed_moving, affine = model(moving, fixed)
        loss = criterion(transformed_moving, fixed)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        tqdm_traindata.set_postfix({'Training Loss': running_loss / (i + 1)})
        end_data = time.time()
        print(f"Time for batch {i + 1}: {end_data - start_data:.4f} seconds")
    
    running_loss /= len(traindata)
    loss_values.append(running_loss)
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}")
model.eval()

test_loss = 0.0
with torch.no_grad():
    for i, data in enumerate(test_data, 0):
        moving = data[0].float().to(device)
        fixed = data[1].float().to(device)
        
        # Forward pass
        transformed_moving, affine = model(moving, fixed)
        
        # Compute loss
        loss = criterion(transformed_moving, fixed)
        test_loss += loss.item()
    
# Calculate average test loss
test_loss /= len(test_data)
print(f"Test Loss: {test_loss:.4f}")
        



    
    
    
        
        
        
        
    


