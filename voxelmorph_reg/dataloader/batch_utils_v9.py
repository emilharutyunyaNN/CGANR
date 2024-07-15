
import os
import h5py
import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import random
from itertools import cycle
import time
#from timm.data.loader import MultiEpochsDataLoader, _RepeatSampler
"""
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
        self.shuffle_time = 4
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
        return self.generator_wrapper()

    def generator_wrapper(self):
        i = 0
        for path in  self.path_iter:
            #print("----", path)
            try:
                
                patch_list = []
                i= 0 
                path = next(self.path_iter)
                generator = HDF5Generator(path, self.hdf5_path, self.config) #TODO images are not shuffled 
                for img, lab in generator:
                # t_trans = time.time()
                    if i==6: # next image time
                        break
                    if self.transform:
                        # print("***")
                        img, lab = self.transform(img, lab)
                    patch_list.append((img, lab))
                    #yield img, lab
                    i+=1
                random.shuffle(patch_list)
                for _ in patch_list:
                    yield _
            except StopIteration:
                break
            #gen_end = time.time()
            #print("generation for one path: ", gen_end - gen_s)
            
        #print("done")

"""
import os
import h5py
import numpy as np
import cv2
import random
import time

class Config:
    def __init__(self):
        self.image_size = 256
        self.filter_blank = True
        self.filter_threshold = 0.1
        self.q_limit = 10

def look_into_hdf5(hdf5_file, dataset_type, config):
    idx = hdf5_file.rfind("/")
    save_dir = hdf5_file[:idx]
    os.makedirs(save_dir, exist_ok=True)
    new_dataset = os.path.join(save_dir, dataset_type)
    os.makedirs(new_dataset, exist_ok=True)
    paths = []

    with h5py.File(hdf5_file, 'r') as hf:
        for class_name in hf[dataset_type].keys():
            new_classname = os.path.join(new_dataset, class_name)
            os.makedirs(new_classname, exist_ok=True)
            for ds_name in hf[dataset_type][class_name].keys():
                new_name = os.path.join(new_classname, ds_name[:-4])
                os.makedirs(new_name, exist_ok=True)
                paths.append(f"{dataset_type}/{class_name}/{ds_name}")
                    
    #print(paths)
    s = config.image_size
    stride = s

    for path in paths:
        save_path = os.path.join(save_dir, path[:-4])
        if "input" in save_path:
            input_save = save_path
            target_save = save_path.replace("input", "target")
        else:
            input_save = save_path.replace("target", "input")
            target_save = save_path    
        
        with h5py.File(hdf5_file, 'r') as hdf5:
            dataset_type, class_name, ds_name = path.split('/')
            data = hdf5[dataset_type][class_name][ds_name][()]
            
            if isinstance(data, np.ndarray):
                if "input" in class_name:
                    image = data[:, :, 0:3].copy() 
                    data_target = hdf5[dataset_type][class_name.replace("input", "target")][ds_name][()]
                    label = data_target[:, :, 0:3].copy() 
                else:
                    label = data[:, :, 0:3].copy() 
                    data_input = hdf5[dataset_type][class_name.replace("target", "input")][ds_name][()]
                    image = data_input[:, :, 0:3].copy() 
            else:
                raise TypeError(f"Expected data to be a numpy array, but got {type(data)}")

            crop_edge = 30
            image = image[crop_edge:-crop_edge, crop_edge:-crop_edge, :]
            label = label[crop_edge:-crop_edge, crop_edge:-crop_edge, :]

            size = min(image.shape[0], image.shape[1])
            time_crop = time.time()
            s = 256
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
                        print(img.shape, lab.shape)
                        if config.filter_blank and np.mean(lab) >= config.filter_threshold and cur_trial_count < config.q_limit:
                            cur_trial_count += 1
                            continue
                        else:
                            cv2.imwrite(os.path.join(input_save, f"{xx}_{yy}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                            cv2.imwrite(os.path.join(target_save, f"{xx}_{yy}.jpg"), cv2.cvtColor(lab, cv2.COLOR_RGB2BGR))

                    if yy == size - s:
                        break
                    y += stride
                if xx == size - s:
                    break
                x += stride

# Example usage
config = Config()
look_into_hdf5('/home/hkhz/daihui.hdf5', 'Validation', config)
look_into_hdf5('/home/hkhz/daihui.hdf5', 'Training', config)