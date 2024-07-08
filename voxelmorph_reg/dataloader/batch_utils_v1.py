import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import cv2
import os
import numpy as np
import random
from scipy.io import loadmat
import matplotlib.pyplot as plt
class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch
class Config:
    def __init__(self, is_training=True, img_size=256, label_channels=3, batch_size=4, n_threads=4, filter_blank=False, filter_threshold=None, channel_start=0, channel_end=3):
        self.is_training = is_training
        self.image_size = img_size
        self.label_channels = label_channels
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.filter_blank = filter_blank
        self.threshold = filter_threshold
        self.channel_start_index = channel_start
        self.channel_end_index = channel_end

class PatchDatasetMini(Dataset):
    def __init__(self, dir, config, transform=None):
        super(PatchDatasetMini, self).__init__()
        self.dir = dir
        self.transform = transform
        self.config = config
        self.imgs = os.listdir(dir)
        self.data_iterator = self._patch_generator()
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        return next(self.data_iterator)

    def _patch_generator(self):
        while True:
            img = random.choice(self.imgs)
            full_path = os.path.join(self.dir, img)
            image = self.parse_and_generate(full_path)

            s = self.config.image_size
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
                        img_patch = image[xx:xx + s, yy:yy + s]
                        #lab_patch = label[xx:xx + s, yy:yy + s]

                        if self.config.filter_blank and np.mean(img_patch) >= self.config.threshold and cur_trial_count < 10:
                            cur_trial_count += 1
                            continue
                        else:
                            yield img_patch.astype(np.float32)#, lab_patch.astype(np.float32)

                    if yy == size - s:
                        break
                    y += s
                if xx == size - s:
                    break
                x += s

    def parse_and_generate(self, path):
        s = self.config.image_size
        start, end = self.config.channel_start_index, self.config.channel_end_index
        """if path.endswith('.mat'):
            image = loadmat(self.config.convert_inp_path_from_target(path))['input'].astype(np.float32)[:, :, start:end]
            label = loadmat(path)['target'].astype(np.float32)
        else:
            image = np.transpose(np.load(self.config.convert_inp_path_from_target(path)).astype(np.float32)[:, :, start:end], axes=[1, 2, 0])
            label = np.transpose(np.load(path).astype(np.float32), axes=[1, 2, 0])
        """
        image = cv2.imread(path) /255.0
        """if self.config.data_inpnorm == 'norm_by_specified_value':
            normalize_vector = [1500, 1500, 1500, 1000]
            normalize_vector = np.reshape(normalize_vector, [1, 1, 4])
            image = image / normalize_vector
        elif self.config.data_inpnorm == 'norm_by_mean_std':
            image = (image - np.mean(image)) / (np.std(image) + 1e-5)
        """
        crop_edge = 30
        image = image[crop_edge:-crop_edge, crop_edge:-crop_edge, :]
        #label = label[crop_edge:-crop_edge, crop_edge:-crop_edge, :]

        return image

    def augment(self, img, lab):
        if self.transform:
            #img, lab = self.transform(img, lab)
            img = self.transform(img)
        return img#, lab

config = Config(is_training=True, img_size=256, label_channels=3, batch_size=1, n_threads=4, filter_blank=False, filter_threshold=None)
dataset = PatchDatasetMini("./data", config=config)

dataloader = InfiniteDataLoader(dataset=dataset, shuffle=config.is_training, batch_size=config.batch_size, num_workers=config.n_threads, pin_memory=True)
import time
iter_dataloader = iter(dataloader)
while True:
    data = next(iter_dataloader)
    img = data
    img = img.numpy()[0]
    #lab = lab.numpy()[0]

    # Plot the image and label
    #plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image")
    """plt.subplot(1, 2, 2)
    plt.imshow(lab)
    plt.title("Label")"""
    plt.show()
    time.sleep(2)
    plt.close()
    #time.sleep(10)

    
