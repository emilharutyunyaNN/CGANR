"""import argparse
import h5py
import numpy as np
from PIL import Image

def print_structure(hdf5_file):
    def print_group(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Class: {obj.attrs['class'] if 'class' in obj.attrs else 'N/A'}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    hdf5_file.visititems(print_group)

def main(args):
    hf = h5py.File(args.hdf5file, 'r')

    print("HDF5 File Structure:")
    print_structure(hf)

    print("===")
    if args.get_dataset and args.get_filename:
        try:
            f = hf[args.get_dataset][args.get_filename]
            image_data = np.array(f)

            # Determine the correct mode for the image
            if image_data.ndim == 3 and image_data.shape[2] in [1, 3, 4]:
                mode = {1: "L", 3: "RGB", 4: "RGBA"}[image_data.shape[2]]
                img = Image.fromarray(image_data, mode)
            else:
                img = Image.fromarray(image_data)

            cls = f.attrs['class']
            print('image size:', img.size, 'class:', cls)
            img.save("./img.jpg")  # Display the image

        except KeyError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read HDF5')
    parser.add_argument('hdf5file', help='Path to the HDF5 file')
    parser.add_argument('get_dataset', nargs='?', help='Name of the dataset group to retrieve')
    parser.add_argument('get_filename', nargs='?', help='Name of the file within the dataset group to retrieve')
    args = parser.parse_args()
    main(args)"""
    
"""import argparse
import h5py
import argparse
import h5py
import io
import numpy as np
from PIL import Image

def print_structure(hdf5_file):
    def print_group(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Class: {obj.attrs['class'] if 'class' in obj.attrs else 'N/A'}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {name}")

    hdf5_file.visititems(print_group)

def main(args):
    hf = h5py.File(args.hdf5file, 'r')

    print("HDF5 File Structure:")
    print_structure(hf)

    print("===")
    if args.get_dataset and args.get_filename:
        try:
            f = hf[args.get_dataset][args.get_filename]
            image_data = np.array(f)
            mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(image_data.shape[2], "RGB")
            img = Image.fromarray(image_data, mode)
            cls = f.attrs['class']
            print('image size:', img.size, 'class:', cls)
            img.show()  # Display the image
        except KeyError as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read HDF5')
    parser.add_argument('hdf5file', help='Path to the HDF5 file')
    parser.add_argument('get_dataset', nargs='?', help='Name of the dataset group to retrieve')
    parser.add_argument('get_filename', nargs='?', help='Name of the file within the dataset group to retrieve')
    args = parser.parse_args()
    main(args)"""
    
import h5py
import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import random
import os

"""class Config:
    def __init__(self):
        self.num_epoch = 10
        self.is_training = True
        self.image_size = 256
        self.channel_start_index = 0
        self.channel_end_index = 3
        self.label_channels = 1
        self.data_inpnorm = 'norm_by_mean_std'
        self.filter_blank = True
        self.filter_threshold = 0.1
        self.batch_size = 32
        self.n_threads = 4
        self.num_slices = 3
        self.case_trial_limit = 10

class InterleavedHDF5Dataset(IterableDataset):
    def __init__(self, hdf5_path, dataset_type, config, transform=None):
        self.hdf5_path = hdf5_path
        self.dataset_type = dataset_type
        self.config = config
        self.transform = transform

        with h5py.File(hdf5_path, 'r') as hf:
            self.paths = []
            for group_name in hf[dataset_type].keys():
                group = hf[dataset_type][group_name]
                #print(group)
                parts = group_name.split('.')
                self.paths.append('/'.join(parts[-3:]))
                if isinstance(group, h5py.Group):
                    for key in group.keys():
                        #print(key)
                        self.paths.append(f"{dataset_type}/{group_name}/{key}")
        #print(self.paths)        
                #self.paths.append(group)
        self.PATHS = []
        for i in range(config.num_epoch):
            if config.is_training:
                random.shuffle(self.paths)
            self.PATHS.extend(self.paths)

    def __len__(self):
        return len(self.PATHS)

    def __iter__(self):
        return self.create_dataset_from_generator()

    def create_dataset_from_generator(self):
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            for path in self.PATHS:
                for img, lab in self.parse_and_generate(hdf5_file, path):
                    if self.transform:
                        img, lab = self.transform(img, lab)
                    yield img, lab

    def parse_and_generate(self, hdf5_file, path):
        s = self.config.image_size
        stride = s

        start, end = self.config.channel_start_index, self.config.channel_end_index

        # Read the image and label data from HDF5
        dataset_name, sub_group, unique_name = path.split('/')
        data = hdf5_file[dataset_name][sub_group][unique_name][()]
        
        print(f"Data type: {type(data)}, Data shape: {data.shape}")
        print(sub_group)
        if "input" in sub_group:
            image = data[:, :, start:end]
            label = data[:, :, end:end + self.config.label_channels]
        else:
            label = data[:, :, start:end]
            image = data[:, :, end:end + self.config.label_channels]

        if self.config.data_inpnorm == 'norm_by_specified_value':
            normalize_vector = [1500, 1500, 1500, 1000]
            normalize_vector = np.reshape(normalize_vector, [1, 1, 4])
            image = image / normalize_vector
        elif self.config.data_inpnorm == 'norm_by_mean_std':
            image = (image - np.mean(image)) / (np.std(image) + 1e-5)

        crop_edge = 30
        image = image[crop_edge:-crop_edge, crop_edge:-crop_edge, :]
        label = label[crop_edge:-crop_edge, crop_edge:-crop_edge, :]

        size = image.shape[0]

        cur_trial_count = 0
        x = 0
        while True:
            y = 0
            while True:
                rand_choice_stride = random.randint(0, 15)
                xx = min(x + rand_choice_stride * s // 16, size - s)
                yy = min(y + rand_choice_stride * s // 16, size - s)
                if yy != size - s and xx != size - s:
                    img = image[xx:xx + s, yy:yy + s]
                    lab = label[xx:xx + s, yy:yy + s]

                    if self.config.filter_blank and np.mean(lab) >= self.config.filter_threshold \
                            and cur_trial_count < self.config.case_trial_limit:
                        cur_trial_count += 1
                        continue
                    else:
                        yield (img.astype(np.float32), lab.astype(np.float32))

                if yy == size - s:
                    break
                y += stride
            if xx == size - s:
                break
            x += stride

class Augmentation:
    def __init__(self, num_slices, label_channels):
        self.num_slices = num_slices
        self.label_channels = label_channels

    def __call__(self, img, lab):
        imglab = np.concatenate([img, lab], axis=-1)
        if random.random() > 0.5:
            imglab = np.flip(imglab, axis=1)

        k = random.randint(0, 3)
        imglab = np.rot90(imglab, k=k)

        img = imglab[:, :, :self.num_slices]
        lab = imglab[:, :, self.num_slices:self.num_slices + self.label_channels]

        return img, lab

# Example configuration
config = Config()

# Initialize the dataset and dataloader for training
transform = Augmentation(num_slices=config.channel_end_index, label_channels=config.label_channels)
train_dataset = InterleavedHDF5Dataset('/home/hkhz/daihui.hdf5', 'Training', config, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.n_threads)

# Initialize the dataset and dataloader for validation
val_dataset = InterleavedHDF5Dataset('/home/hkhz/daihui.hdf5', 'Validation', config, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.n_threads)

# Example training loop
for batch_data, batch_labels in train_dataloader:
    # Your training code here
    print('Training:', batch_data.shape, batch_labels.shape)

# Example validation loop
for batch_data, batch_labels in val_dataloader:
    # Your validation code here
    print('Validation:', batch_data.shape, batch_labels.shape)
"""
import os
import cv2
import h5py
import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import random

class Config:
    def __init__(self):
        self.num_epoch = 10
        self.is_training = True
        self.image_size = 256
        self.channel_start_index = 0
        self.channel_end_index = 3
        self.label_channels = 3
        self.data_inpnorm = 'norm_by_mean_std'
        self.filter_blank = True
        self.filter_threshold = 0.1
        self.batch_size = 32
        self.n_threads = 4
        self.num_slices = 3
        self.case_trial_limit = 10

class Augmentation:
    def __init__(self, num_slices, label_channels):
        self.num_slices = num_slices
        self.label_channels = label_channels

    def __call__(self, img, lab, path, patch_id):
        imglab = np.concatenate([img, lab], axis=-1)

        # Save original input and label images for debugging
        self.save_debug_image(img, path, patch_id, "concat_input")
        self.save_debug_image(lab, path, patch_id, "concat_label")

        if random.random() > 0.5:
            imglab = np.flip(imglab, axis=1).copy()  # Ensure contiguous array
            # Save flipped input and label images for debugging
            self.save_debug_image(imglab[:, :, :self.num_slices], path, patch_id, "flip_input")
            self.save_debug_image(imglab[:, :, self.num_slices:], path, patch_id, "flip_label")

        k = random.randint(0, 3)
        imglab = np.rot90(imglab, k=k).copy()  # Ensure contiguous array
        # Save rotated input and label images for debugging
        self.save_debug_image(imglab[:, :, :self.num_slices], path, patch_id, "rotate_input")
        self.save_debug_image(imglab[:, :, self.num_slices:], path, patch_id, "rotate_label")

        img = imglab[:, :, :self.num_slices]
        lab = imglab[:, :, self.num_slices:self.num_slices + self.label_channels]

        # Save augmented images
        self.save_debug_image(img, path, patch_id, "aug_img")
        self.save_debug_image(lab, path, patch_id, "aug_lab")

        return img, lab

    def save_debug_image(self, img, path, patch_id, stage):
        # Ensure the directory exists
        os.makedirs(f"augmented/debug/{stage}", exist_ok=True)
        debug_path = f"augmented/debug/{stage}/{path.replace('/', '_')}_{patch_id}.png"
        cv2.imwrite(debug_path, cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

class InterleavedHDF5Dataset(IterableDataset):
    def __init__(self, hdf5_path, dataset_type, config, transform=None):
        self.hdf5_path = hdf5_path
        self.dataset_type = dataset_type
        self.config = config
        self.transform = transform

        self.paths = []
        with h5py.File(hdf5_path, 'r') as hf:
            for class_name in hf[dataset_type].keys():
                for ds_name in hf[dataset_type][class_name].keys():
                    self.paths.append(f"{dataset_type}/{class_name}/{ds_name}")

        print(f"Total paths found: {len(self.paths)}")
        self.PATHS = []
        for i in range(config.num_epoch):
            if config.is_training:
                random.shuffle(self.paths)
            self.PATHS.extend(self.paths)

    def __len__(self):
        return len(self.PATHS)

    def __iter__(self):
        return self.create_dataset_from_generator()

    def create_dataset_from_generator(self):
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            for path in self.PATHS:
                for img, lab, patch_id in self.parse_and_generate(hdf5_file, path):
                    # Save cropped patches before augmentation
                    self.save_patch(img, lab, path, patch_id, "cropped")
                    if self.transform:
                        img, lab = self.transform(img, lab, path, patch_id)
                    yield img, lab

    def parse_and_generate(self, hdf5_file, path):
        s = self.config.image_size
        stride = s

        start, end = self.config.channel_start_index, self.config.channel_end_index

        dataset_type, class_name, ds_name = path.split('/')
        data = hdf5_file[dataset_type][class_name][ds_name][()]

        if isinstance(data, np.ndarray):
            if "input" in class_name:
                image = data[:, :, start:end].copy()/255.0  # Ensure contiguous array
                data_target = hdf5_file[dataset_type][class_name.replace("input", "target")][ds_name][()]
                label = data_target[:, :, start:end].copy()/255.0  # Ensure contiguous array
            else:
                label = data[:, :, start:end].copy()/255.0  # Ensure contiguous array
                data_input = hdf5_file[dataset_type][class_name.replace("target", "input")][ds_name][()]
                image = data_input[:, :, start:end].copy()/255.0  # Ensure contiguous array
        else:
            raise TypeError(f"Expected data to be a numpy array, but got {type(data)}")

        if self.config.data_inpnorm == 'norm_by_specified_value':
            normalize_vector = [1500, 1500, 1500]
            normalize_vector = np.reshape(normalize_vector, [1, 1, 3])
            image = image / normalize_vector
        elif self.config.data_inpnorm == 'norm_by_mean_std':
            image = (image - np.mean(image)) / (np.std(image) + 1e-5)

        crop_edge = 30
        image = image[crop_edge:-crop_edge, crop_edge:-crop_edge, :]
        label = label[crop_edge:-crop_edge, crop_edge:-crop_edge, :]

        size = min(image.shape[0], image.shape[1])

        patch_count = 0
        x = 0
        while True:
            y = 0
            while True:
                rand_choice_stride = random.randint(0, 15)
                xx = min(x + rand_choice_stride * s // 16, size - s)
                yy = min(y + rand_choice_stride * s // 16, size - s)
                if yy != size - s and xx != size - s:
                    img = image[xx:xx + s, yy:yy + s]
                    lab = label[xx:xx + s, yy:yy + s]
                    patch_count+=1
                    #("sending: ", img.shape, lab.shape, xx, yy, s, image.shape, label.shape)
                    yield {'input': img.astype(np.float32), 'target': lab.astype(np.float32)}
                if yy == size - s:
                    break
                y += stride
            if xx == size - s:
                break
            x += stride

    def save_patch(self, img, lab, path, patch_id, stage):
        # Ensure the directory exists
        os.makedirs(f"augmented/{stage}", exist_ok=True)
        # Save the patches using OpenCV
        img_path = f"augmented/{stage}/{path.replace('/', '_')}_{patch_id}_img.png"
        lab_path = f"augmented/{stage}/{path.replace('/', '_')}_{patch_id}_lab.png"
        cv2.imwrite(img_path, cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(lab_path, cv2.cvtColor((lab * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

# Example configuration
config = Config()

# Initialize the dataset and dataloader for training
transform = Augmentation(num_slices=config.channel_end_index, label_channels=config.label_channels)
train_dataset = InterleavedHDF5Dataset('/home/hkhz/daihui.hdf5', 'Training', config, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.n_threads)

# Initialize the dataset and dataloader for validation
val_dataset = InterleavedHDF5Dataset('/home/hkhz/daihui.hdf5', 'Validation', config, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.n_threads)

# Example training loop
for batch_data, batch_labels in train_dataloader:
    # Your training code here
    print('Training:', batch_data.shape, batch_labels.shape)

# Example validation loop
for batch_data, batch_labels in val_dataloader:
    # Your validation code here
    print('Validation:', batch_data.shape, batch_labels.shape)
