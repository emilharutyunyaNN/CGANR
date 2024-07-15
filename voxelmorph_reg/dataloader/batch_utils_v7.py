import os
import h5py
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import Dataset
import numpy as np
import random
import itertools
from torch.utils.data import get_worker_info
import gc
import time
from itertools import cycle

class Config:
    def __init__(self):
        self.N_epoch = 10
        self.is_training = True
        self.image_size = 256
        self.channel_start_index = 0
        self.channel_end_index = 3
        self.label_channels = 3
        self.data_inpnorm = 'norm_by_mean_std'
        self.filter_blank = True
        self.filter_threshold = 0.1
        self.batch_size = 4
        self.n_threads = 4
        self.num_slices = 3
        self.q_limit = 10

class Augmentation:
    def __init__(self, num_slices, label_channels):
        self.num_slices = num_slices
        self.label_channels = label_channels

    def __call__(self, img, lab):
        imglab = np.concatenate([img, lab], axis=0)

        if random.random() > 0.5:
            imglab = np.flip(imglab, axis=2).copy()

        k = random.randint(0, 3)
        imglab = np.rot90(imglab, k=k, axes=(1, 2)).copy()

        img = imglab[:self.num_slices]
        lab = imglab[self.num_slices:self.num_slices + self.label_channels]

        return img, lab

class HDF5Generator:
    def __init__(self, path, hdf5_path, config):
        self.path = path
        self.hdf5_path = hdf5_path
        self.config = config

    def __iter__(self):
        return self._generator()

    def _generator(self):
        s = self.config.image_size
        stride = s

        start, end = self.config.channel_start_index, self.config.channel_end_index

        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            dataset_type, class_name, ds_name = self.path.split('/')
            data = hdf5_file[dataset_type][class_name][ds_name][()]

            if isinstance(data, np.ndarray):
                if "input" in class_name:
                    image = data[:, :, start:end].copy() / 255.0
                    data_target = hdf5_file[dataset_type][class_name.replace("input", "target")][ds_name][()]
                    label = data_target[:, :, start:end].copy() / 255.0
                else:
                    label = data[:, :, start:end].copy() / 255.0
                    data_input = hdf5_file[dataset_type][class_name.replace("target", "input")][ds_name][()]
                    image = data_input[:, :, start:end].copy() / 255.0
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

            cur_trial_count = 0
            x = 0
            while x < size:
                y = 0
                while y < size:
                    rand_choice_stride = random.randint(0, 15)
                    xx = min(x + rand_choice_stride * s // 16, size - s)
                    yy = min(y + rand_choice_stride * s // 16, size - s)
                    if yy != size - s and xx != size - s:
                        img = image[xx:xx + s, yy:yy + s]
                        lab = label[xx:xx + s, yy:yy + s]

                        if self.config.filter_blank and np.mean(lab) >= self.config.filter_threshold \
                                and cur_trial_count < self.config.q_limit:
                            cur_trial_count += 1
                            continue
                        else:
                            yield (img.transpose(2, 0, 1).astype(np.float32), lab.transpose(2, 0, 1).astype(np.float32))

                    if yy == size - s:
                        break
                    y += stride
                if xx == size - s:
                    break
                x += stride

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
        #self.dataset = Dataset.from_list(self.paths)
        print(f"Total paths found: {len(self.paths)}")
        
        if self.config.is_training:
            random.shuffle(self.paths)
        self.path_iter = cycle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        return self.generator_wrapper()

    def generator_wrapper(self):
        while True:
            path_batch = list(itertools.islice(self.path_iter, self.config.n_threads))
            if not path_batch:
                break
            if self.config.is_training:
                random.shuffle(path_batch)
            for path in path_batch:
                generator = HDF5Generator(path, self.hdf5_path, self.config)
                for img, lab in generator:
                    if self.transform:
                        img, lab = self.transform(img, lab)
                    yield img, lab

class InfiniteDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

    def reset(self):
        self.iterator = iter(self.dataloader)

# Example usage
"""if __name__ == "__main__":
    config = Config()
    train_images = ["path/to/train_image1.h5", "path/to/train_image2.h5"]
    transform = Augmentation(config.num_slices, config.label_channels)
    train_dataset = InterleavedHDF5Dataset("path/to/hdf5_file.h5", "Training", config, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.n_threads)
    infinite_train_loader = InfiniteDataLoader(train_loader)

    for epoch in range(config.N_epoch):
        epoch_start_time = time.time()
        for i, (inputs, targets) in enumerate(infinite_train_loader):
            # Training code here
            pass
        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} duration: {epoch_end_time - epoch_start_time:.2f} seconds")
        infinite_train_loader.reset()  """# Reset the loader at the end of each epoch

"""
if __name__ == "__main__":
    config = Config()
    transform = Augmentation(num_slices=config.channel_end_index, label_channels=config.label_channels)

    train_dataset = InterleavedHDF5Dataset('/path/to/dataset.hdf5', 'Training', config, transform=transform)
    val_dataset = InterleavedHDF5Dataset('/path/to/dataset.hdf5', 'Validation', config, transform=None)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.n_threads, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.n_threads, pin_memory=True)

    infinite_train_dataloader = InfiniteDataLoader(train_dataloader)
    infinite_val_dataloader = InfiniteDataLoader(val_dataloader)

    for epoch in range(config.N_epoch):
        print(f"Epoch {epoch + 1}")

        for i in range(100):  # Example iteration count
            data_t = next(infinite_train_dataloader)
            input_images, targets = data_t
            # Training steps here
            print(f"Train Iter {i}: Input shape: {input_images.shape}, Target shape: {targets.shape}")

        # Validation step
        for j in range(10):  # Example iteration count
            data_v = next(infinite_val_dataloader)
            valid_x, valid_y = data_v
            # Validation steps here
            print(f"Valid Iter {j}: Input shape: {valid_x.shape}, Target shape: {valid_y.shape}")
"""

# Example configuration
#config = Config()

"""# Initialize the dataset and dataloader for training
transform = Augmentation(num_slices=config.channel_end_index, label_channels=config.label_channels)

# Create training and validation dataset instances
train_dataset = InterleavedHDF5Dataset('/home/hkhz/daihui.hdf5', 'Training', config, transform=transform)
val_dataset = InterleavedHDF5Dataset('/home/hkhz/daihui.hdf5', 'Validation', config, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.n_threads)
val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.n_threads)

# Infinite dataloaders
infinite_train_dataloader = InfiniteDataLoader(train_dataloader)
infinite_val_dataloader = InfiniteDataLoader(val_dataloader)"""
"""
iter = 0
while True:
    iter += 1
    # Example usage of next functionality
    batch = next(infinite_train_dataloader)
    print("Batch contents: ", batch)
    print("Batch length: ", len(batch))
    print("Batch element type: ", type(batch[0]))
    print("Batch element length: ", len(batch[0]))

    # Correct way to unpack and process the batch
    batch_data, batch_labels = batch
    batch_data_f = np.clip(batch_data[0].numpy().astype(np.float32), 0, 1)
    batch_labels_f = np.clip(batch_labels[0].numpy().astype(np.float32), 0, 1)

    # Plotting only the first channel for gray mode and normal mode for labels
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(batch_data_f[0], cmap='gray')
    plt.title('Batch Data - Channel 0')

    plt.subplot(1, 2, 2)
    plt.imshow(batch_labels_f.transpose(1, 2, 0))
    plt.title('Batch Labels')

    plt.savefig('./fig_batch.jpg')
    plt.show()

    print('Training:', batch_data.shape, batch_labels.shape)
    if iter == 100:
        break

# Example validation loop
for batch in infinite_val_dataloader:
    # Debugging
    print("Batch contents: ", batch)

    # Correct way to unpack and process the batch
    batch_data, batch_labels = batch
    print("Batch data shape: ", [b.shape for b in batch_data])
    print("Batch labels shape: ", [b.shape for b in batch_labels])

    # Your validation code here
    print('Validation:', batch_data.shape, batch_labels.shape)
    # For demonstration purposes, we break after the first batch
    break
"""