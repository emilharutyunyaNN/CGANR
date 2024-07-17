import os
import h5py
import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import random
from itertools import cycle
import matplotlib.pyplot as plt

class HDF5Reader:
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path

    def get_image_pairs(self, dataset_type):
        pairs = []
        with h5py.File(self.hdf5_path, 'r') as hf:
            input_group_name = f"{dataset_type}/input_{dataset_type.lower()}"
            target_group_name = f"{dataset_type}/target_{dataset_type.lower()}"
            input_group = hf[input_group_name]
            target_group = hf[target_group_name]
            
            for x in input_group.keys():
                for img_name in input_group[x].keys():
                    input_path = f"{input_group_name}/{x}/{img_name}"
                    target_path = f"{target_group_name}/{x}/{img_name}"
                    pairs.append((input_path, target_path))
                
        return pairs

class HDF5IterableDataset(IterableDataset):
    def __init__(self, config, transform=None):
        self.config = config
        self.hdf5_reader = HDF5Reader(self.config.hdf5_path)
        
        self.dataset_type = self.config.dataset_type
        
        self.transform = transform
        self.pairs = self.hdf5_reader.get_image_pairs(self.dataset_type)
        if self.config.is_training:
            random.shuffle(self.pairs)
        self.pairs_iter = cycle(self.pairs)  # Create an indefinite iterator

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # In a single-worker setup
            self.pairs_iter = cycle(self.pairs)
        else:  # In a multi-worker setup
            per_worker = int(np.ceil(len(self.pairs) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            self.pairs_iter = cycle(self.pairs[worker_id * per_worker: (worker_id + 1) * per_worker])
        return self

    def __next__(self):
        input_path, target_path = next(self.pairs_iter)
        
        with h5py.File(self.hdf5_reader.hdf5_path, 'r') as hf:
            input_img = hf[input_path][()]
            target_img = hf[target_path][()]

        input_img = torch.tensor(input_img).permute(2, 0, 1).float() / 255.0
        
        target_img = torch.tensor(target_img).permute(2, 0, 1).float() / 255.0
        
        if self.config.data_inpnorm == 'norm_by_specified_value':
            normalize_vector = torch.tensor([1500, 1500, 1500]).view(1, 3, 1, 1)
            input_img = input_img / normalize_vector
        elif self.config.data_inpnorm == 'norm_by_mean_std':
            #print("here")
            input_mean = input_img.mean()
            input_std = input_img.std() + 1e-5
            input_img = (input_img - input_mean) / input_std
        
        if self.transform:
            #print("--")
            input_img, target_img = self.transform(input_img, target_img)
        
        return input_img, target_img

    def __getitem__(self, index):
        return self.__next__()

class Augmentation:
    def __init__(self, num_slices, label_channels):
        self.num_slices = num_slices
        self.label_channels = label_channels

    def __call__(self, img, lab):
        imglab = torch.cat([img, lab], axis=0)

        if random.random() > 0.5:
            imglab = torch.flip(imglab, dims=[2])

        k = random.randint(0, 3)
        imglab = torch.rot90(imglab, k=k, dims=[1, 2])

        img = imglab[:self.num_slices]
        lab = imglab[self.num_slices:self.num_slices + self.label_channels]

        return img, lab  

def save_images(input_imgs, target_imgs, epoch, batch_idx = None):
    input_img = np.clip(input_imgs[0][0].cpu().numpy(),0,1)  # Grayscale, first channel of the first image in the batch
    target_img = target_imgs[0].permute(1, 2, 0).cpu().numpy()  # First image in the batch
    print(np.min(input_img))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    
    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title(f'Epoch {epoch}, Batch {batch_idx}, Input')
    axes[0].axis('off')
    
    axes[1].imshow(target_img)
    axes[1].set_title(f'Epoch {epoch}, Batch {batch_idx}, Target')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'fig.png')
    plt.close()
import time

def worker_init_fn(worker_id):
    random.seed(worker_id)
    np.random.seed(worker_id)
    
    
from configobj import ConfigObj
def init_parameters():
    tc, vc = ConfigObj(), ConfigObj()
    tc.hdf5_path = '/home/hkhz/data_daihui.hdf5'
    vc.hdf5_path = '/home/hkhz/data_daihui.hdf5'
    tc.model_path = '/home/hkhz/emil/Project_ai/voxelmorph_reg/models' # set the path to save model
    tc.prev_checkpoint_path = None
    tc.save_every_epoch = True

    # pretrained checkpoints to start from
    tc.G_warmstart_checkpoint = None
    tc.D_warmstart_checkpoint = None
    tc.R_warmstart_checkpoint = None
    assert not (tc.prev_checkpoint_path
                and (tc.G_warmstart_checkpoint or tc.D_warmstart_checkpoint or tc.R_warmstart_checkpoint))

    #tc.image_path = "/home/hkhz/daihui/Training/target/*.mat" # path for training data 
    #vc.image_path = "/home/hkhz/daihui/Validation/target/*.mat" # path for validation data

    #tc.image_path = "/home/hkhz/daihui/Training/target/*.jpg"
    #vc.image_path = "/home/hkhz/daihui/Validation/target/*.jpg"
    tc.dataset_type = "Training"
    vc.dataset_type = "Validation"
    def convert_inp_path_from_target(inp_path: str):
        return inp_path.replace('target', 'input')

    tc.convert_inp_path_from_target = convert_inp_path_from_target
    vc.convert_inp_path_from_target = convert_inp_path_from_target

    tc.is_mat, vc.is_mat =False, False #True, True  # True for .mat, False for .npy
    #tc.is_mat,vc.is_mat = False, False
    tc.data_inpnorm, vc.data_inpnorm = 'norm_by_mean_std', 'norm_by_mean_std'
    tc.channel_start_index, vc.channel_start_index = 0, 0
    tc.channel_end_index, vc.channel_end_index = 3, 3  # exclusive

    # network and loss params
    tc.is_training, vc.is_training = True, False
    tc.image_size, vc.image_size = 256, 256
    tc.num_slices, vc.num_slices = 3, 3
    tc.label_channels, vc.label_channels = 3, 3
    assert tc.channel_end_index - tc.channel_start_index == tc.num_slices
    assert vc.channel_end_index - vc.channel_start_index == vc.num_slices
    tc.n_channels, vc.n_channels = 32, 32
    tc.lamda = 50.0  # adv loss

    tc.nf_enc, vc.nf_enc = [8, 16, 16, 32, 32], [8, 16, 16, 32, 32]  # for aligner
    tc.nf_dec, vc.nf_dec = [32, 32, 32, 32, 32, 16, 16], [32, 32, 32, 32, 32, 16, 16]  # for aligner
    tc.R_loss_type = 'ncc'
    tc.lambda_r_tv = 1.0  # .1    # tv of predicted flow
    tc.gauss_kernel_size = 80
    tc.n_threads, vc.n_threads= 2, 2
    tc.dvf_clipping = True  # clip DVF to [mu-sigma*dvf_clipping_nsigma, mu+sigma*dvf_clipping_nsigma]
    tc.dvf_clipping_nsigma = 3
    tc.dvf_thresholding = True  # clip DVF to [-dvf_thresholding_distance, dvf_thresholding_distance]
    tc.dvf_thresholding_distance = 30

    # training params
    tc.batch_size, vc.batch_size = 4, 4
    tc.n_shuffle_epoch, vc.n_shuffle_epoch = 500, 5000  # for the batchloader
    tc.initial_alternate_steps = 6000  # train G/D for initial_alternate_steps steps before switching to R for the same # of steps
    tc.valid_steps = 100  # perform validation when D_steps % valid_steps == 0 or at the end of a loop of (train G/D, train R)
    tc.n_threads, vc.n_threads = 6, 6
    tc.q_limit, vc.q_limit = 100, 300
    tc.N_epoch = 20  # number of loops
    vc.N_epoch = 20
    tc.tol = 0  # current early stopping patience
    tc.max_tol = 2  # the max-allowed early stopping patience
    tc.min_del = 0  # the lowest acceptable loss value reduction

    # case filtering
    tc.case_filtering = False
    tc.case_filtering_metric = 'ncc'  # 'ncc'
    # divide each patch into case_filtering_x_subdivision patches alone the x dimension for filtering (1 = no division)
    tc.case_filtering_x_subdivision = 2
    tc.case_filtering_y_subdivision = 2
    assert tc.case_filtering_x_subdivision >= 1 and tc.case_filtering_y_subdivision >= 1
    tc.case_filtering_starting_epoch = 2  # case filtering only when epoch >= case_filtering_starting_epoch
    tc.case_filtering_cur_mean, tc.case_filtering_cur_stdev = 0.3757, 0.0654  # for lung elastic (256x256 patch)
    tc.case_filtering_nsigma = 2
    tc.case_filtering_recalc_every_eval = True

    # case filtering for dataloader
    tc.filter_blank, vc.filter_blank = True, True
    tc.filter_threshold, vc.filter_threshold = 0.9515, 0.9515  # 0.9515 for elastic/MT

    # per-pixel loss mask to account for out of the field information brought in by R
    tc.loss_mask, vc.loss_mask = False, False  # True, False

    # training resume parameters
    tc.epoch_begin = 0
    # this overrides tc.epoch_  begin the training schedule; tc.epoch_begin is required for logging
    # set it to None when not used
    tc.iter_begin =  None

    return tc, vc
    
"""   
    
def main():
    hdf5_path = '/home/hkhz/data_daihui.hdf5'
    #dataset_type = 'Training'  # or 'Validation'
    tc,vc = init_parameters()
    # Define augmentation
    num_slices = 3  # Adjust based on your data
    label_channels = 3  # Adjust based on your data
    augmentation = Augmentation(num_slices, label_channels)

    # Create dataset
    dataset_train = HDF5IterableDataset(config=tc, transform=augmentation)
    dataset_val = HDF5IterableDataset(config=vc, transform=None)
    # Use DataLoader to load the dataset
    dataloader_train = DataLoader(dataset_train, batch_size=4, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=4, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    
    
    iter_d_train = iter(dataloader_train)
    iter_d_val = iter(dataloader_val)
    # Iterate over the dataset using DataLoader with a for loop
    for epoch in range(10):  
        iter_d_train = iter(dataloader_train)
        iter_d_val = iter(dataloader_val)# Example: 4 epochs
        #for batch_idx, (input_imgs, target_imgs) in enumerate(dataloader):
         #   print(f"Epoch {epoch}, Batch {batch_idx}: input shape {input_imgs[0].shape}, target shape {target_imgs[0].shape}")
        input_imgs, target_imgs = next(iter_d_train)    
        save_images(input_imgs, target_imgs, epoch)
        #time.sleep(1)
        
    for epoch in range(5):
        input_imgs, target_imgs = next(iter_d_val)
        print(input_imgs.shape, target_imgs.shape)

if __name__ == "__main__":
    main()
"""