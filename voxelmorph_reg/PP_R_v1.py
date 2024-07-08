import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from networks import VxmDense, default_unet_features
from losses import NCC
from torch.nn import functional as F
#from MLP_check_v4 import MLP_resnet, Residual

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Data processing with torch dataset and dataloader



"""
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
    def __init__(self, data, registration_model=None, transform=None):
        self.data = data
        self.registration_model = registration_model
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_bank, img_rgb = self.data[idx]
        
        if self.registration_model:
            image_bank, img_rgb = self.apply_registration(image_bank, img_rgb)
        
        pixel_values, locations, y = self.preprocess_images(image_bank, img_rgb)
        
        if self.transform:
            pixel_values = self.transform(pixel_values)
            locations = self.transform(locations)
            y = self.transform(y)
        print(pixel_values.shape)
        print(y.shape)
        return {'pixel_values': pixel_values, 'locations': locations, 'y': y}

    def apply_registration(self, image_bank, img_rgb):
        moving_image = torch.tensor(image_bank, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        fixed_image = torch.tensor(img_rgb, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)

        with torch.no_grad():
            moved_image, _ = self.registration_model(moving_image, fixed_image)
        
        moved_image = moved_image.squeeze().cpu().numpy()
        fixed_image = fixed_image.squeeze().cpu().numpy()
        
        return moved_image, fixed_image

    def preprocess_images(self, image_bank, img_rgb):
        height, width = image_bank.shape[1], image_bank.shape[2]
        pixel_values = image_bank.reshape(len(image_bank), -1).T
        locations = np.array([(h, w) for h in range(height) for w in range(width)])
        y = img_rgb.reshape(-1, 3)
        return pixel_values, locations, y

# Swap training function
def swap_training(network_to_train, network_to_not_train):
    for param in network_to_not_train.parameters():
        param.requires_grad = False

    for param in network_to_train.parameters():
        param.requires_grad = True

    network_to_not_train.eval()
    network_to_train.train()
#MODELS
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
        self.resnet = Residual(300, 128, 1, 0, True)
        self.fc1 = nn.Linear(128, 3)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1).permute(0, 2, 1)
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x




# Define your networks
model_registration = VxmDense(
    inshape=(784, 480),
    nb_unet_features=default_unet_features(),
    nb_unet_levels=None,
    unet_feat_mult=1,
    nb_unet_conv_per_level=1,
    int_steps=7,
    int_downsize=2,
    bidir=False,
    use_probs=False,
    src_feats=1,
    trg_feats=1,
    unet_half_res=False
).to(device)

model_classification = MLP_resnet().to(device)

# Losses and optimizers
image_loss_fn = NCC().loss
regularization_fn = nn.MSELoss()
criterion_classification = nn.CrossEntropyLoss()

optimizer_registration = torch.optim.Adam(model_registration.parameters(), lr=1e-5)
optimizer_classification = torch.optim.Adam(model_classification.parameters(), lr=0.0001, weight_decay=0.001)

# Load and prepare data
data = load_data()
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = ImagePairDataset(data=train_data, registration_model=model_registration, transform=transform)
val_dataset = ImagePairDataset(data=val_data, registration_model=model_registration, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# Alternating training loop
num_iterations = 5
num_epochs_per_iteration = 10

for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}/{num_iterations}")

    # Step 1: Train the Registration Network
    swap_training(model_registration, model_classification)
    for epoch in range(num_epochs_per_iteration):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            fixed_image = batch['y'].to(device)
            moving_image = batch['pixel_values'].to(device)

            optimizer_registration.zero_grad()
            moved_image, flow = model_registration(moving_image, fixed_image)
            loss = image_loss_fn(moved_image, fixed_image) + regularization_fn(flow, torch.zeros_like(flow))
            loss.backward()
            optimizer_registration.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 batches
                print(f"[Iteration {iteration + 1}, Epoch {epoch + 1}, Batch {i + 1}] Registration loss: {running_loss / 10:.3f}")
                running_loss = 0.0

    # Apply the trained registration model to align images for regression
    aligned_images = []
    model_registration.eval()
    with torch.no_grad():
        for batch in val_loader:
            fixed_image = batch['y'].to(device)
            moving_image = batch['pixel_values'].to(device)
            moved_image, _ = model_registration(moving_image, fixed_image)
            aligned_images.append((moved_image.cpu().numpy(), fixed_image.cpu().numpy()))

    # Step 2: Train the Classification Network using the aligned images
    swap_training(model_classification, model_registration)
    for epoch in range(num_epochs_per_iteration):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer_classification.zero_grad()
            outputs = model_classification(inputs)
            loss = criterion_classification(outputs, targets)
            loss.backward()
            optimizer_classification.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 batches
                print(f"[Iteration {iteration + 1}, Epoch {epoch + 1}, Batch {i + 1}] Classification loss: {running_loss / 10:.3f}")
                running_loss = 0.0

        # Evaluate the classification network
        model_classification.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['pixel_values'].to(device)
                targets = batch['y'].to(device)
                outputs = model_classification(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print(f"Iteration {iteration + 1}, Epoch {epoch + 1}, Classification Accuracy: {100 * correct / total:.2f}%")

print("Training completed.")
"""
"""import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from networks import VxmDense, default_unet_features
from losses import NCC
from torch.nn import functional as F
import multiprocessing

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
    def __init__(self, data, registration_model=None, transform=None):
        self.data = data
        self.registration_model = registration_model
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_bank, img_rgb = self.data[idx]
        
        if self.registration_model:
            image_bank = self.apply_registration(image_bank, img_rgb)
        
        pixel_values, y = self.extract_features(image_bank, img_rgb)
        
        if self.transform:
            pixel_values = self.transform(pixel_values)
            y = self.transform(y)
        
        return {'pixel_values': pixel_values, 'y': y}

    def apply_registration(self, image_bank, img_rgb):
        registered_bank = []
        fixed_image = torch.tensor(img_rgb, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        
        for img in image_bank:
            moving_image = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                moved_image, _ = self.registration_model(moving_image, fixed_image)
            registered_bank.append(moved_image.squeeze().cpu().numpy())
        
        return np.array(registered_bank)

    def extract_features(self, image_bank, img_rgb):
        height, width = image_bank.shape[1], image_bank.shape[2]
        pixel_values = image_bank.reshape(len(image_bank), -1).T
        y = img_rgb.reshape(-1, 3)
        return pixel_values, y

# Swap training function
def swap_training(network_to_train, network_to_not_train):
    for param in network_to_not_train.parameters():
        param.requires_grad = False

    for param in network_to_train.parameters():
        param.requires_grad = True

    network_to_not_train.eval()
    network_to_train.train()

# Define your networks
model_registration = VxmDense(
    inshape=(784, 480),
    nb_unet_features=default_unet_features(),
    nb_unet_levels=None,
    unet_feat_mult=1,
    nb_unet_conv_per_level=1,
    int_steps=7,
    int_downsize=2,
    bidir=False,
    use_probs=False,
    src_feats=1,
    trg_feats=1,
    unet_half_res=False
).to(device)

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
        self.resnet = Residual(300, 128, 1, 0, True)
        self.fc1 = nn.Linear(128, 3)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1).permute(0, 2, 1)
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Define your networks
model_registration = VxmDense(
    inshape=(784, 480),
    nb_unet_features=default_unet_features(),
    nb_unet_levels=None,
    unet_feat_mult=1,
    nb_unet_conv_per_level=1,
    int_steps=7,
    int_downsize=2,
    bidir=False,
    use_probs=False,
    src_feats=1,
    trg_feats=1,
    unet_half_res=False
).to(device)

model_classification = MLP_resnet().to(device)

# Losses and optimizers
criterion_registration = NCC().loss
optimizer_registration = optim.Adam(model_registration.parameters(), lr=1e-4)

criterion_classification = nn.MSELoss()
optimizer_classification = optim.Adam(model_classification.parameters(), lr=1e-4)

# Load and preprocess data
data = load_data()

# Create dataset and dataloaders
dataset = ImagePairDataset(data, registration_model=model_registration)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Train registration model
    swap_training(model_registration, model_classification)
    running_loss_registration = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, targets = data['pixel_values'].to(device), data['y'].to(device)
        optimizer_registration.zero_grad()
        outputs, _ = model_registration(inputs, targets)
        print("---------------")
        loss = criterion_registration(outputs, targets)
        loss.backward()
        optimizer_registration.step()
        running_loss_registration += loss.item()
    print(f"Registration Loss: {running_loss_registration / len(train_loader)}")

    # Train classification model
    swap_training(model_classification, model_registration)
    running_loss_classification = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, targets = data['pixel_values'].to(device), data['y'].to(device)
        optimizer_classification.zero_grad()
        outputs = model_classification(inputs)
        loss = criterion_classification(outputs, targets)
        loss.backward()
        optimizer_classification.step()
        running_loss_classification += loss.item()
    print(f"Classification Loss: {running_loss_classification / len(train_loader)}")

print("Training completed")
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from networks import VxmDense, default_unet_features
from losses import NCC
from torch.nn import functional as F
import multiprocessing
from tqdm import tqdm

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
            img_gray = cv2.imread(img_path, cv2.IMREAD_COLOR)
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
    def __init__(self, data, registration_model=None, transform=None):
        self.data = data
        print(len(data[0]))
        self.registration_model = registration_model
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_bank, img_rgb = self.data[idx]
        #print(image_bank.shape, img_rgb.shape)
        if self.registration_model:
            image_bank = self.apply_registration(image_bank, img_rgb)
        
        pixel_values, y = self.extract_features(image_bank, img_rgb)
        print(pixel_values.shape, y.shape)
        if self.transform:
            pixel_values = self.transform(pixel_values)
            y = self.transform(y)
        
        return pixel_values, y

    def apply_registration(self, image_bank, img_rgb):
        registered_bank = []
        fixed_image = torch.tensor(img_rgb, dtype=torch.float32).unsqueeze(0).to(device)
        
        for img in image_bank:
            moving_image = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
            self.registration_model.eval()
            with torch.no_grad():
                moved_image, _ = self.registration_model(moving_image, fixed_image)
            registered_bank.append(moved_image.squeeze().cpu().numpy())
        
        return np.array(registered_bank)

    def extract_features(self, image_bank, img_rgb):
        img_bank = img_bank[:,:,:,0]
        height, width = image_bank.shape[1], image_bank.shape[2]
        #pixel_values = image_bank.reshape(len(image_bank), -1).T
        pixel_values = image_bank.reshape(-1, height * width) 
        y = img_rgb.reshape(-1, 3).transpose(1,0)
        return pixel_values, y
    def reconstruct_images(self, pixel_values, height, width):
        num_images = pixel_values.shape[1]
        reconstructed_images = []
        for i in range(num_images):
            image = pixel_values[:, i].reshape(height, width)
            reconstructed_images.append(image)
        return np.array(reconstructed_images)
    def reconstruct_images_pred(self, pixel_values, height, width):
        num_pixels = height * width
        rgb_images = pixel_values.reshape(num_pixels, 3).reshape(height, width, 3)
        return rgb_images

# Swap training function
def swap_training(network_to_train, network_to_not_train):
    for param in network_to_not_train.parameters():
        param.requires_grad = False

    for param in network_to_train.parameters():
        param.requires_grad = True

    network_to_not_train.eval()
    network_to_train.train()

# Define your networks


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
        self.resnet = Residual(300, 128, 1, 0, True)
        self.fc1 = nn.Linear(128, 3)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1).permute(0, 2, 1)
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Define your networks
model_registration = VxmDense(
    inshape=(784, 480),
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

model_classification = MLP_resnet().to(device)

# Losses and optimizers
criterion_registration = NCC().loss
optimizer_registration = optim.Adam(model_registration.parameters(), lr=1e-4)

criterion_classification = nn.MSELoss()
optimizer_classification = optim.Adam(model_classification.parameters(), lr=1e-4)

# Load and preprocess data
#data = load_data()
#print(len(data))
data = [
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3)),
        ([np.random.rand(784, 480, 3) for _ in range(300)], np.random.rand(784, 480, 3))
        # Add more pairs here as needed
]
# Create dataset and dataloaders
dataset = ImagePairDataset(data, registration_model=model_registration)
data_l = [len(dataset[i]) for i in range(len(dataset))]
print(len(dataset))
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
print(len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
print(train_loader)
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
     # Train classification model
    swap_training(model_classification, model_registration)
    running_loss_classification = 0.0
   # with tqdm(total=len(train_loader), desc="Training Classification Model") as pbar:
    print("swap done")
    for i, data_p in enumerate(train_loader):
        print("--")
        try:
            print(data_p)
            inputs, targets = data_p[0].to(device), data_p[1].to(device)
            print(inputs.shape)
            optimizer_classification.zero_grad()
            outputs = model_classification(inputs)
            print(outputs.shape)
            loss = criterion_classification(outputs, targets)
            loss.backward()
            optimizer_classification.step()
            print("-----")
            running_loss_classification += loss.item()
            pbar.update(1)
            pbar.set_postfix({"Loss": running_loss_classification / (i + 1)})
        except Exception as e:
            print(f"Exception occurred in iteration {i}: {e}")
        print(f"Classification Loss: {running_loss_classification / len(train_loader)}")
    # Train registration model
    swap_training(model_registration, model_classification)
    running_loss_registration = 0.0
    with tqdm(total=len(train_loader), desc="Training Registration Model") as pbar:
        for i, data in enumerate(train_loader, 0):
            inputs, targets = data['pixel_values'].to(device), data['rgb'].to(device)
            optimizer_registration.zero_grad()
            outputs_initial = model_classification(inputs)
            height, width = targets.shape[0], targets.shape[1]
            reconstructed_img = dataset.reconstruct_images_pred(outputs_initial, height, width)
            reconstructed_img = torch.tensor(reconstructed_img).permute(2, 0, 1).unsqueeze(0).to(device)
            targets = targets.permute(2, 0, 1).unsqueeze(0).to(device)
            outputs, _ = model_registration(reconstructed_img, targets)
            loss = criterion_registration(outputs, targets)
            loss.backward()
            optimizer_registration.step()
            running_loss_registration += loss.item()
            pbar.update(1)
            pbar.set_postfix({"Loss": running_loss_registration / (i + 1)})
    print(f"Registration Loss: {running_loss_registration / len(train_loader)}")

   

print("Training completed")
