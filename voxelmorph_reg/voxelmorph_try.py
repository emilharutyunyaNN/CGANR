import logging
import sys
import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from losses import NCC
# Define Unet and VxmDense as provided previously
# Make sure the networks.py file containing these definitions is in the same directory
from networks import VxmDense, default_unet_features
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""# Logging setup
def log(output_file):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    logger = logging.getLogger()
    while os.path.exists(output_file):
        output_file = output_file.replace('.log', '_1.log')
    if output_file:
        file_handler = logging.FileHandler(output_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)
    return logger

# Set the logger
#logger_monai = log('./log_files/result_36.log')

# Data preparation
def get_data(path_to_data="../results"):
    data = []
    for sub_dir in os.listdir(path_to_data):
        another_sub = os.path.join(path_to_data, sub_dir)
        for i in range(1, 16):
            try:
                idx_str = f'00{i}' if i < 10 else f'0{i}'
                if len(os.listdir(os.path.join(another_sub, idx_str))) > 0 and len(os.listdir(os.path.join(another_sub, f'{idx_str}-{idx_str}'))) > 0:
                    data.append([
                        os.path.join(another_sub, idx_str, os.listdir(os.path.join(another_sub, idx_str))[1]),
                        os.path.join(another_sub, f'{idx_str}-{idx_str}', os.listdir(os.path.join(another_sub, f'{idx_str}-{idx_str}'))[1])
                    ])
            except:
                continue
    return data

def crop_n_save(data):
    os.makedirs('./data_monai', exist_ok=True)
    data_new = []
    for item in data:
        im0 = cv2.imread(item[0])[:808]
        im1 = cv2.imread(item[1])[:808]
        save_path0 = os.path.join('./data_monai', item[0].split(os.sep)[-3], item[0].split(os.sep)[-2])
        save_path1 = os.path.join('./data_monai', item[1].split(os.sep)[-3], item[1].split(os.sep)[-2])
        os.makedirs(save_path0, exist_ok=True)
        os.makedirs(save_path1, exist_ok=True)
        cv2.imwrite(os.path.join(save_path0, 'im0.jpg'), im0)
        cv2.imwrite(os.path.join(save_path1, 'im1.jpg'), im1)
        data_new.append([os.path.join(save_path0, 'im0.jpg'), os.path.join(save_path1, 'im1.jpg')])
    return data_new

img_data = crop_n_save(get_data())

# Data loading and transformations
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fixed_image = cv2.imread(self.data[idx][1], cv2.IMREAD_COLOR)
        moving_image = cv2.imread(self.data[idx][0], cv2.IMREAD_COLOR)
        sample = {'fixed_image': fixed_image, 'moving_image': moving_image}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        fixed_image, moving_image = sample['fixed_image'], sample['moving_image']
        # Pad the images to ensure they have dimensions divisible by 16
        desired_height = (fixed_image.shape[0] // 16 + 1) * 16 if fixed_image.shape[0] % 16 != 0 else fixed_image.shape[0]
        desired_width = (fixed_image.shape[1] // 16 + 1) * 16 if fixed_image.shape[1] % 16 != 0 else fixed_image.shape[1]
        print("size: ", desired_height, desired_width)
        pad_height = desired_height - fixed_image.shape[0]
        pad_width = desired_width - fixed_image.shape[1]

        fixed_image = np.pad(fixed_image, ((0, pad_height), (0, pad_width), (0,0)), mode='constant', constant_values=0)
        moving_image = np.pad(moving_image, ((0, pad_height), (0, pad_width),(0,0)), mode='constant', constant_values=0)

        return {
            'fixed_image': torch.tensor(fixed_image, dtype=torch.float32).permute(2,0,1) / 255.0,
            'moving_image': torch.tensor(moving_image, dtype=torch.float32).permute(2,0,1) / 255.0
        }

transform = transforms.Compose([ToTensor()])
train_dataset = CustomDataset(data=img_data[:100], transform=transform)
val_dataset = CustomDataset(data=img_data[100:134], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# Model setup
model = VxmDense(
    inshape=(816, 480),  # Adjust the inshape to match padded image size
    nb_unet_features=default_unet_features(),
    nb_unet_levels=None,
    unet_feat_mult=1,
    nb_unet_conv_per_level=1,
    int_steps=7,
    int_downsize=2,
    bidir=False,
    use_probs=False,
    src_feats=3,
    trg_feats=3,
    unet_half_res=False
).to(device)
#import pytorch_ssim

#image_loss_fn1 = pytorch_ssim.SSIM(window_size=11)
image_loss_fn = nn.MSELoss()
#image_loss_fn = NCC().loss
regularization_fn = nn.MSELoss()  # or any other regularization loss you want to use

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Training and validation loop
max_epochs = 200
epoch_loss_values = []
val_loss_values = []

for epoch in range(max_epochs):
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        optimizer.zero_grad()
        fixed_image = batch_data["fixed_image"].to(device)
        moving_image = batch_data["moving_image"].to(device)
        print(moving_image.shape)
        print(fixed_image.shape)

        y_source, preint_flow = model(moving_image, fixed_image)
        
        image_loss = image_loss_fn(fixed_image, y_source)
        regularization_loss = regularization_fn(preint_flow, torch.zeros_like(preint_flow))  # Adjust regularization loss as needed
        loss = image_loss + regularization_loss

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    model.eval()
    val_loss = 0
    val_step = 0
    with torch.no_grad():
        for val_data in val_loader:
            val_step += 1
            fixed_image = val_data["fixed_image"].to(device)
            moving_image = val_data["moving_image"].to(device)

            y_source, preint_flow = model(moving_image, fixed_image)
            val_image_loss = image_loss_fn(y_source, fixed_image)
            val_loss += val_image_loss.item()

    val_loss /= val_step
    val_loss_values.append(val_loss)
    print(f"epoch {epoch + 1} validation loss: {val_loss:.4f}")

# Plotting the loss curves
plt.figure()
plt.plot(epoch_loss_values, label='Training Loss')
plt.plot(val_loss_values, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./plot_ncc.jpg')
#plt.show()


torch.save(model.state_dict(), 'trained_model_color.pth')"""

loaded_model = VxmDense(
    inshape=(816, 480),  # Adjust the inshape to match padded image size
    nb_unet_features=default_unet_features(),
    nb_unet_levels=None,
    unet_feat_mult=1,
    nb_unet_conv_per_level=1,
    int_steps=7,
    int_downsize=2,
    bidir=False,
    use_probs=False,
    src_feats=3,
    trg_feats=3,
    unet_half_res=False
).to(device)

loaded_model.load_state_dict(torch.load('trained_model_color.pth'))
loaded_model.eval()

# Define a function to apply the loaded model to transform two images
def transform_images(model, moving_image_path, fixed_image_path):
    moving_image = cv2.imread(moving_image_path, cv2.IMREAD_COLOR)
    fixed_image = cv2.imread(fixed_image_path, cv2.IMREAD_COLOR)
    print(moving_image.shape)
    moving_image = moving_image[:-5,:]
    fixed_image = fixed_image[:-32,:]
    print(fixed_image.shape)
    # Pad the images to ensure they have dimensions divisible by 16
    desired_height = (fixed_image.shape[0] // 16 + 1) * 16 if fixed_image.shape[0] % 16 != 0 else fixed_image.shape[0]
    desired_width = (fixed_image.shape[1] // 16 + 1) * 16 if fixed_image.shape[1] % 16 != 0 else fixed_image.shape[1]
    pad_height = desired_height - fixed_image.shape[0]
    pad_width = desired_width - fixed_image.shape[1]
    fixed_image = np.pad(fixed_image, ((0, pad_height), (0, pad_width),(0,0)), mode='constant', constant_values=0)
    moving_image = np.pad(moving_image, ((0, pad_height), (0, pad_width), (0,0)), mode='constant', constant_values=0)
    print(fixed_image.shape)
    print(moving_image.shape)
    # Convert images to tensors
    fixed_image_tensor = torch.tensor(fixed_image, dtype=torch.float32).permute(2,0,1) / 255.0
    moving_image_tensor = torch.tensor(moving_image, dtype=torch.float32).permute(2,0,1) / 255.0
    print(fixed_image_tensor.shape)
    print(moving_image_tensor.shape)
    # Move tensors to the appropriate device
    fixed_image_tensor = fixed_image_tensor.unsqueeze(0).to(device)
    moving_image_tensor = moving_image_tensor.unsqueeze(0).to(device)

    # Apply the model to transform the moving image
    transformed_image, _ = model(moving_image_tensor, fixed_image_tensor)

    # Convert the transformed tensor back to a numpy array
    transformed_image_np = transformed_image.squeeze(0).squeeze(0).cpu().detach().numpy() * 255.0
    transformed_image_np = transformed_image_np.astype(np.uint8)

    return transformed_image_np

# Example usage
transformed_image = transform_images(loaded_model, '../results/07-0003/009/rgb4.jpg', '../results/07-0003/009-009/rgb4.jpg')

# Save the transformed image
cv2.imwrite('./transformed_image.jpg', transformed_image)
       
