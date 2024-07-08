import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
import multiprocessing
from tqdm import tqdm
from torch.nn import functional as F

# Set multiprocessing start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the data loading and processing functions
def load_and_process_image(path_rgb, band_files_dir):
    try:
        img_rgb = cv2.imread(path_rgb, cv2.IMREAD_COLOR)
        if img_rgb is None:
            raise ValueError(f"Failed to read RGB image: {path_rgb}")
        
        img_rgb = img_rgb[img_rgb.shape[0] - 784 - 29: -29] / 255.0
        image_bank = []
        
        img_files = sorted(os.listdir(band_files_dir))
        for img_file in img_files:
            img_path = os.path.join(band_files_dir, img_file)
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                raise ValueError(f"Failed to read grayscale image: {img_path}")
            image_bank.append(img_gray[:784] / 255.0)
        
        return np.array(image_bank), img_rgb
    except Exception as e:
        print(f"Error processing images: {e}")
        return None

def process_image_set(dir1, dir2, dir, i):
    if i < 10:
        path_rgb = os.path.join(dir2, dir, f'00{i}-00{i}/rgb4.jpg')
        band_files_dir = os.path.join(dir1, dir, f'00{i}')
    else:
        path_rgb = os.path.join(dir2, dir, f'0{i}-0{i}/rgb4.jpg')
        band_files_dir = os.path.join(dir1, dir, f'0{i}')

    if not os.path.exists(path_rgb) or not os.path.exists(band_files_dir):
        print(f"Paths do not exist: {path_rgb}, {band_files_dir}")
        return None

    return load_and_process_image(path_rgb, band_files_dir)

def process_batch(dir1, dir2, dir, batch):
    results = []
    with ThreadPoolExecutor() as executor:
        future_to_params = {
            executor.submit(process_image_set, dir1, dir2, dir, i): (dir, i)
            for i in batch
        }

        for future in as_completed(future_to_params):
            params = future_to_params[future]
            try:
                result = future.result()
                if result is not None:
                    print(f"Appending result from set {params[1]} in directory {params[0]}")
                    results.append(result)
            except Exception as e:
                print(f"Exception occurred while processing {params}: {e}")

    return results

def load_data(dir1='/home/yqb/dataset/dataset/image/temp2', dir2='/home/yqb/dataset/results', batch_size=12):
    all_data = []
    sub_dirs = os.listdir(dir1)

    for dir in sub_dirs:
        print(f"Processing directory: {dir}")
        batches = [list(range(i, min(i + batch_size, 16))) for i in range(1, 16, batch_size)]
        
        for batch in batches:
            results = process_batch(dir1, dir2, dir, batch)
            all_data.extend(results)

    return all_data

# Define Custom Dataset and DataLoader
class ImagePairDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data) * 784 * 480  # Total number of pixels in all images

    def __getitem__(self, idx):
        img_idx = idx // (784 * 480)
        pixel_idx = idx % (784 * 480)
        
        image_bank, img_rgb = self.data[img_idx]
        
        # Get pixel coordinates
        h = pixel_idx // 480
        w = pixel_idx % 480
        
        # Pixel values from all 300 channels
        pixel_values = image_bank[:, h, w]
        
        # Corresponding RGB values
        y = img_rgb[h, w, :]
        
        if self.transform:
            pixel_values = self.transform(pixel_values)
            y = self.transform(y)
        
        return torch.from_numpy(pixel_values).float(), torch.from_numpy(y).float()

# Define the MLP model with residual connections
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        if kernel_size == 3:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=strides)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=strides)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, X):
        Y = F.relu((self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
            X = self.bn2(X)
        Y += X
        return F.relu(Y)

class MLP_resnet(nn.Module):
    def __init__(self):
        super(MLP_resnet, self).__init__()
        self.fc1 = nn.Linear(300, 128)
        self.fc2 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define the classification model
model_classification = MLP_resnet().to(device)

# Define the loss function and optimizer
criterion_classification = nn.MSELoss()
optimizer_classification = optim.Adam(model_classification.parameters(), lr=1e-4)

# Load and preprocess data
data = load_data()
print(f"Total images loaded: {len(data)}")

# Create dataset and dataloaders
dataset = ImagePairDataset(data)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model_classification.train()
    running_loss_classification = 0.0

    for i, (inputs, targets) in enumerate(train_loader):
        try:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer_classification.zero_grad()
            
            outputs = model_classification(inputs)
            loss = criterion_classification(outputs, targets)
            loss.backward()
            optimizer_classification.step()
            
            running_loss_classification += loss.item()
            
            if i % 100 == 99:  # Print every 100 batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss_classification / 100:.3f}")
                running_loss_classification = 0.0
        except Exception as e:
            print(f"Error in training loop: {e}")

    # Validation loop
    model_classification.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_classification(inputs)
            loss = criterion_classification(outputs, targets)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    print(f"Validation loss after epoch {epoch + 1}: {val_loss:.3f}")

print("Training complete.")

# Reconstruction
def reconstruct_image(model, data):
    model.eval()
    reconstructed_images = []
    with torch.no_grad():
        for img_data in data:
            image_bank, _ = img_data
            image_bank = torch.from_numpy(image_bank).float().to(device)
            
            reconstructed_img = np.zeros((784, 480, 3), dtype=np.float32)
            for h in range(784):
                for w in range(480):
                    pixel_values = image_bank[:, h, w].unsqueeze(0)
                    rgb_values = model(pixel_values).cpu().numpy().squeeze()
                    reconstructed_img[h, w, :] = rgb_values
            reconstructed_images.append(reconstructed_img)
    return reconstructed_images

# Reconstruct images
reconstructed_images = reconstruct_image(model_classification, data)

# Save or visualize reconstructed images
# For example, if using OpenCV to save images:
for i, img in enumerate(reconstructed_images):
    cv2.imwrite(f"reconstructed_image_{i}.jpg", (img * 255).astype(np.uint8))