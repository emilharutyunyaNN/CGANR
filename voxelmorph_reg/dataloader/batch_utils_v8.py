#TODO: in tf from_generator loads the data lazily during iteration by specified batch size not all at once, interleave puts them in a deetermenistic sequence which is decided by  block length and cycle length



import os
import h5py
import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import random
from itertools import cycle
import time
#from timm.data.loader import MultiEpochsDataLoader, _RepeatSampler

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
        time_op = time.time()
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
            time_crop = time.time()
            #print("cropped in : ", time_crop - time_op)
            cur_trial_count = 0
            x = 0
            #print("image size: ", size)
            while x < size:
                #print("x: ", x)
                y = 0
                while y < size:
                    #print("y: ", y)
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
                            #print("yielding time: ", time.time() - time_op)
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
        #path loading
        #path_start = time.time()
        with h5py.File(hdf5_path, 'r') as hf:
            for class_name in hf[dataset_type].keys():
                for ds_name in hf[dataset_type][class_name].keys():
                    self.paths.append(f"{dataset_type}/{class_name}/{ds_name}")
        #path_end = time.time()
       # print("time to load paths: ", path_end -path_start)
        #print(f"Total paths found: {len(self.paths)}")
        if self.config.is_training:
            random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        if self.config.is_training:
            random.shuffle(self.paths)  # Shuffle paths at the start of each epoch
        self.path_iter = cycle(self.paths)
        #self.pt = self.path_iter
        #print("----", len(list(self.path_iter)))# Reinitialize the cycle iterator with shuffled paths
        return self.generator_wrapper() # next in training code is called on this and this returns the yield in the geneerator

    def generator_wrapper(self):
        i = 0
        for path in self.path_iter:
            #print("----", path)
           
            generator = HDF5Generator(path, self.hdf5_path, self.config) #TODO images are not shuffled 
            #gen_s = time.time()
            
            for img, lab in generator:
               # t_trans = time.time()
                if self.transform:
                   # print("***")
                    img, lab = self.transform(img, lab)
                #e_trans = time.time()
                #print("transformation time: ", e_trans - t_trans)
                #print(img.shape, lab.shape)
                yield img, lab
            #gen_end = time.time()
            #print("generation for one path: ", gen_end - gen_s)
            
        #print("done")


#TODO: fix dataloader
#TODO: RUN model in docker, also affine


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
        
        
class MultiEpochsDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __iter__(self):
        for _ in self.iterator:
            yield _

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# Example usage
"""if __name__ == "__main__":
    config = Config()

    # Initialize the dataset and dataloader for training
    transform = Augmentation(num_slices=config.channel_end_index, label_channels=config.label_channels)

    # Create training and validation dataset instances
    train_dataset = InterleavedHDF5Dataset('/home/hkhz/daihui.hdf5', 'Training', config, transform=transform)
    val_dataset = InterleavedHDF5Dataset('/home/hkhz/daihui.hdf5', 'Validation', config, transform=transform)

    train_dataloader = MultiEpochsDataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.n_threads)
    val_dataloader = MultiEpochsDataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.n_threads)

    # Infinite dataloaders
    #infinite_train_dataloader = InfiniteDataLoader(train_dataloader)
    #infinite_val_dataloader = InfiniteDataLoader(val_dataloader)

    # Testing data loading
    iter_train = iter(train_dataloader)
    iter_val = iter(val_dataloader)

    for _ in range(5):
        batch = next(iter_train)
        print("Batch shape:", [b.shape for b in batch])"""
