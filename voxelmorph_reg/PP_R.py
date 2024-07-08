import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from networks import VxmDense, default_unet_features
from losses import NCC
from ..MLP_check_v3 import MLP_resnet, load_data, data_preprocessing, CustomDataset
from voxelmorph_try import ToTensor, crop_n_save, get_data
from sklearn.model_selection import train_test_split

# Assuming VxmDense and MLP_resnet are already defined as in the previous parts

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def swap_training(network_to_train, network_to_not_train):
    for param in network_to_not_train.parameters():
        param.requires_grad = False

    for param in network_to_train.parameters():
        param.requires_grad = True

    network_to_not_train.eval()
    network_to_train.train()

# Define your networks
model_registration = VxmDense(
    inshape=(816, 480),
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
# CustomDataset, ToTensor, get_data, crop_n_save, etc. should be defined as in previous parts

transform = transforms.Compose([ToTensor()])
img_data = crop_n_save(get_data())

train_dataset = CustomDataset(data=img_data[:100], transform=transform)
val_dataset = CustomDataset(data=img_data[100:134], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# Classification data preprocessing
data, ground_truth, image_bank = load_data(dir1='./ground_truth/transform_check')
X, Y = data_preprocessing(data)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

train_dataset_classification = CustomDataset(np.concatenate((X_train, Y_train[:, np.newaxis]), axis=1))
test_dataset_classification = CustomDataset(np.concatenate((X_test, Y_test[:, np.newaxis]), axis=1))

train_loader_classification = DataLoader(train_dataset_classification, batch_size=32, shuffle=True)
test_loader_classification = DataLoader(test_dataset_classification, batch_size=32, shuffle=False)

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
            fixed_image = batch['fixed_image'].to(device)
            moving_image = batch['moving_image'].to(device)

            optimizer_registration.zero_grad()
            moved_image, flow = model_registration(moving_image, fixed_image)
            loss = image_loss_fn(moved_image, fixed_image) + regularization_fn(flow, torch.zeros_like(flow))
            loss.backward()
            optimizer_registration.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 batches
                print(f"[Iteration {iteration + 1}, Epoch {epoch + 1}, Batch {i + 1}] Registration loss: {running_loss / 10:.3f}")
                running_loss = 0.0

    # Apply the trained registration model to align images for classification
    aligned_images = []
    model_registration.eval()
    with torch.no_grad():
        for batch in val_loader:
            fixed_image = batch['fixed_image'].to(device)
            moving_image = batch['moving_image'].to(device)
            moved_image, _ = model_registration(moving_image, fixed_image)
            aligned_images.append(moved_image.cpu().numpy())

    # Step 2: Train the Classification Network using the aligned images
    swap_training(model_classification, model_registration)
    for epoch in range(num_epochs_per_iteration):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader_classification):
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
            for inputs, targets in test_loader_classification:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_classification(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print(f"Iteration {iteration + 1}, Epoch {epoch + 1}, Classification Accuracy: {100 * correct / total:.2f}%")

print("Training completed.")
