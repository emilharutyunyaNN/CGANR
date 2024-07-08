"""import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from itertools import cycle, islice, repeat
import cv2
import os
import numpy as np
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
    def __init__(self, is_training = True, img_size = 256, label_channels = 3, batch_size = 4, n_threads = 4, filter_blank = False, filter_threshold = None, channel_start= 0, channel_end = 3):
        self.is_training = is_training
        self.image_size = img_size
        self.label_channels = label_channels
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.filter_blank = filter_blank
        self.threshold = filter_threshold
        self.channel_start_index = channel_start
        self.channel_end_index = channel_end

import random

class PatchDatasetMini(Dataset):
    def __init__(self, dir, config = None, transfrom = None):
        super(PatchDatasetMini, self).__init__()
        self.dir = dir
        self.transform = transfrom
        self.config = config
        self.imgs = os.listdir(dir)
        #self.generator = self.parse_and_generate()
        #self.data_iterator = cycle(self.imgs)
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img = self.imgs[index]
        full_path = os.path.join(self.dir, img)
        img = self.parse_and_generate(full_path)

        #image = cv2.imread(full_path, cv2.IMREAD_COLOR)/255.0
        #image = torch.tensor(image, dtype=torch.float32)

    
        return torch.tensor(next(img), dtype=torch.float32).permute(2,0,1)
        return next(self.generator)
    def parse_and_generate(self, path):
        s = self.config.image_size
        stride = s
        start, end = self.config.channel_start_index, self.config.channel_end_index

        image = cv2.imread(path, cv2.IMREAD_COLOR)/255.0

        crop_edge = 30

        image = image[crop_edge:-crop_edge, crop_edge:-crop_edge, :]
        size = image.shape[0]

        current_trial_count = 0
        x = 0
        while True:
            y = 0
            while True:
                random_choice_stride = random.randint(0,15)
                xx = min(x+random_choice_stride*s//16, size-s)
                yy = min(y+random_choice_stride*s//16, size-s)
                if yy!= size-s and xx !=size-s:
                    img = image[xx:xx+s, yy:yy+s]
                    if self.config.filter_blank and np.mean(img)>=self.config.filter_threshold and current_trial_count < self.case_trial_limit:
                        current_trial_count+=1
                        continue
                    else:
                        yield img.astype(np.float32)
                if yy==size-s:
                    break
                y+=stride
            if xx==size-s:
                break
            x+=stride


def custom_collate_fn(batch):
    batch = [item for sublist in batch for item in sublist]
    return torch.stack(batch)

config = Config(is_training = True, img_size = 256, label_channels = 3, batch_size = 1, n_threads = 4, filter_blank = False, filter_threshold = None)
dataset = PatchDatasetMini("./data", config = config)

dataloader = InfiniteDataLoader(dataset=dataset, shuffle = config.is_training,batch_size=config.batch_size,num_workers=config.n_threads, pin_memory=True, collate_fn =custom_collate_fn)

iter_dataloader = iter(dataloader)
while True:
    data = next(iter_dataloader)
    img = data.numpy()[0]

    plt.imshow(img)
    plt.show()
    plt.close()
    
    

"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random

class InfiniteDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        self.dataset = dataset
        self.batch_size = kwargs.get('batch_size', 1)
        self.num_workers = kwargs.get('num_workers', 0)
        self.shuffle = kwargs.get('shuffle', True)
        super().__init__(dataset=self.dataset, *args, **kwargs)
        self.dataset_iterator = self._get_iterator()

    def __iter__(self):
        return self

    def _get_iterator(self):
        while True:
            """if self.shuffle:
                sampler = torch.utils.data.RandomSampler(self.dataset, replacement=True)
            else:
                sampler = torch.utils.data.SequentialSampler(self.dataset)"""
            loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)
            for batch in loader:
                yield batch

    def __next__(self):
        return next(self.dataset_iterator)


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
    def __init__(self, dir, config=None, transform=None):
        super(PatchDatasetMini, self).__init__()
        self.dir = dir
        self.transform = transform
        self.config = config
        self.imgs = os.listdir(dir)
        self.case_trial_limit = 10

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_name = self.imgs[index]
        full_path = os.path.join(self.dir, img_name)
        img = next(self.parse_and_generate(full_path))
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1),img_name

    def parse_and_generate(self, path):
        s = self.config.image_size
        stride = s
        start, end = self.config.channel_start_index, self.config.channel_end_index

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            return
        image = image / 255.0

        crop_edge = 30
        image = image[crop_edge:-crop_edge, crop_edge:-crop_edge, :]
        size = image.shape[0]

        current_trial_count = 0
        x = 0
        while True:
            y = 0
            while True:
                random_choice_stride = random.randint(0, 15)
                xx = min(x + random_choice_stride * s // 16, size - s)
                yy = min(y + random_choice_stride * s // 16, size - s)
                if yy != size - s and xx != size - s:
                    img = image[xx:xx + s, yy:yy + s]
                    if self.config.filter_blank and np.mean(img) >= self.config.filter_threshold and current_trial_count < self.case_trial_limit:
                        current_trial_count += 1
                        continue
                    else:
                        print(xx, "------------------",yy, ": ", path, "-------------", x, y)
                        yield img.astype(np.float32)
                if yy == size - s:
                    break
                y += stride
            if xx == size - s:
                break
            x += stride


config = Config(is_training=True, img_size=256, label_channels=3, batch_size=1, n_threads=4, filter_blank=False, filter_threshold=None)
dataset = PatchDatasetMini("./data", config=config)

dataloader = InfiniteDataLoader(dataset=dataset, shuffle=config.is_training, batch_size=config.batch_size, num_workers=0, pin_memory=True)

iter_dataloader = iter(dataloader)
iter = 0
while True:
    data = next(iter_dataloader)
    #print(data[0])
    img, name = data[0][0].permute(1, 2, 0).numpy(), data[1][0]
    
    plt.imshow(img)
    plt.savefig(f"./verify/{name}_{iter}.jpg")
    iter+=1
