from GAN_torch.att_unet_2d import att_unet_2d
from GAN_torch.discriminator_2d import discriminator_2d
from GAN_torch.losses_for_gan import *
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import torch.functional as F
from networks import VxmDense
from configobj import ConfigObj
import os
import time
from tqdm import tqdm
import cv2
#from concurrent.futures import ThreadPoolExecutor, as_completed
#import multiprocessing
from sklearn.model_selection import train_test_split
#multiprocessing.set_start_method('spawn', force=True)
# Define device
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
import tracemalloc

def start_tracing():
    tracemalloc.start()

def stop_tracing():
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[top 10 memory usage]")
    for stat in top_stats[:10]:
        print(stat)

def print_memory_usage_by_function():
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics('traceback')
    for stat in stats[:10]:
        print(stat)






import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import concurrent.futures
import time
from sklearn.model_selection import train_test_split

print(torch.__version__)
logging.basicConfig(level=logging.INFO)


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTERPORT"] = 12355
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    destroy_process_group() 

"""class PatchDataset(Dataset):
    def __init__(self, dir1, dir2, patch_size=256, stride=128, transform=None):
        self.dir1 = dir1
        self.dir2 = dir2
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.image_paths = self.get_image_paths()

    def get_image_paths(self):
        image_paths = []
        sub_dirs = os.listdir(self.dir1)
        for dir in sub_dirs:
            for i in range(1, 16):
                path_rgb = os.path.join(self.dir2, dir, f'{i:03d}-{i:03d}/rgb4.jpg')
                band_files_dir = os.path.join(self.dir1, dir, f'{i:03d}')
                if os.path.exists(path_rgb) and os.path.exists(band_files_dir):
                    image_paths.append((path_rgb, band_files_dir))
                else:
                    logging.warning(f"Missing path_rgb: {path_rgb} or band_files_dir: {band_files_dir}")
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path_rgb, band_files_dir = self.image_paths[idx]
        try:
            img_bank, img_rgb = self.load_and_process_image(path_rgb, band_files_dir)
            print("shape: ", img_bank[0].shape)
        except Exception as e:
            logging.error(f"Error loading image {path_rgb} or {band_files_dir}: {e}")
            raise e
        
        logging.info(f"Successfully loaded image {path_rgb} and {band_files_dir}")

        rgb_patches = self.extract_patches(img_rgb)
        band_patches_list = [self.extract_patches(band) for band in img_bank]

        patches = []
        for i in range(len(rgb_patches)):
            rgb_patch = rgb_patches[i]
            band_patches = [band_patches[i] for band_patches in band_patches_list]
            patches.append((rgb_patch, band_patches))

        if self.transform:
            # Transform each band rgb
            patches = [(self.transform(rgb), [self.transform(band) for band in bands]) for rgb, bands in patches]

        rgb_patches = torch.stack([torch.from_numpy(rgb).permute(2, 0, 1).float() for rgb, _ in patches])
        # Extract only one channel of the loaded hyperspectral
        band_patches = torch.stack([torch.stack([torch.from_numpy(band[:, :, 0]).float() for band in bands], dim=0) for _, bands in patches])
        print("band shape ", band_patches.shape)
        return rgb_patches, band_patches

    def load_and_process_image(self, path_rgb, band_files_dir):
        start_time = time.time()

        img_rgb = cv2.imread(path_rgb, cv2.IMREAD_COLOR)
        if img_rgb is None:
            raise ValueError(f"Failed to read RGB image: {path_rgb}")
        img_rgb = img_rgb[img_rgb.shape[0] - 784 - 29: -29] / 255.0

        image_bank = self.read_band_images(band_files_dir)

        end_time = time.time()
        logging.info(f"Time taken for load_and_process_image: {end_time - start_time:.4f} seconds")

        return np.array(image_bank), img_rgb

    def read_band_images(self, band_files_dir):
        img_files = sorted(os.listdir(band_files_dir))
        image_bank = []
        for img_file in img_files:
            img_path = os.path.join(band_files_dir, img_file)
            img_gray = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_gray is None:
                raise ValueError(f"Failed to read grayscale image: {img_path}")
            image_bank.append(img_gray[:784] / 255.0)
        return image_bank

    def extract_patches(self, image):
        patches = []
        h, w = image.shape[:2]
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                patch = image[i:i + self.patch_size, j:j + self.patch_size]
                patches.append(patch)
        # print("num of patches: ", len(patches)) #loads three patches
        return patches"""

class PatchDataset(Dataset):
    def __init__(self, dir1, dir2,indices = None, patch_size=256, stride=256, transform=None):
        self.dir1 = dir1
        self.dir2 = dir2
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.image_paths = self.get_image_paths()
        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]

    def get_image_paths(self):
        image_paths = []
        filenames = os.listdir(self.dir1)
        for filename in filenames:
            path_stained = os.path.join(self.dir1, filename)
            path_not_stained = os.path.join(self.dir2, 'not_'+filename)
            #band_files_dir = os.path.join(self.dir1, dir, f'{i:03d}')
            if os.path.exists(path_stained) and os.path.exists(path_not_stained):
                image_paths.append((path_stained, path_not_stained))
            else:
                logging.warning(f"Missing path_rgb: {path_stained} or band_files_dir: {path_not_stained}")
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        stained_image_p, not_stained_image_p = self.image_paths[idx]
        try:
            img_stained, img_not_stained = self.load_and_process_image(stained_image_p, not_stained_image_p)
            #print("shape: ", img_bank[0].shape)
        except Exception as e:
            logging.error(f"Error loading image {stained_image_p} or {not_stained_image_p}: {e}")
            raise e
        
        logging.info(f"Successfully loaded image {stained_image_p} and {not_stained_image_p}")

        stained_patches = self.extract_patches(img_stained) # each 256x256 from original 1024x1024 image
        not_stained_patches = self.extract_patches(img_not_stained) #same with not stained images

        patches = []
        for i in range(len(stained_patches)):
            stained_patch = stained_patches[i]
            #print("**", stained_patch.shape)
            not_stained_patch = not_stained_patches[i]
            #print("++", not_stained_patch.shape)
            patches.append((stained_patch, not_stained_patch))
        #print(patches[0][0])
        if self.transform:
            # Transform each band rgb
            patches = [(self.transform(pair[0]), self.transform(pair[1])) for pair in patches]
        #print("shape patch: ", patches[0][0].shape, patches[0][1].shape)
        stained_patches = torch.stack([torch.from_numpy(stained).permute(2, 0, 1).float() for stained, _ in patches])
        not_stained_patches = torch.stack([torch.from_numpy(not_stained).permute(2, 0, 1).float() for _, not_stained in patches])

        # Extract only one channel of the loaded hyperspectral
        #band_patches = torch.stack([torch.stack([torch.from_numpy(band[:, :, 0]).float() for band in bands], dim=0) for _, bands in patches])
        #rint("band shape ", band_patches.shape)
        #return rgb_patches, band_patches
        return stained_patches, not_stained_patches

    def load_and_process_image(self, path_stained, path_not_stained):
        start_time = time.time()

        img_stained = cv2.imread(path_stained, cv2.IMREAD_COLOR)
        img_not_stained = cv2.imread(path_not_stained, cv2.IMREAD_COLOR)
        #print("Image shapes: ", img_stained.shape, img_not_stained.shape)
        if img_stained is None or img_not_stained is None:
            raise ValueError(f"Failed to read RGB image: {path_stained} or {path_not_stained}")
        #img_rgb = img_rgb[img_rgb.shape[0] - 784 - 29: -29] / 255.0

        #image_bank = self.read_band_images(band_files_dir)

        end_time = time.time()
        logging.info(f"Time taken for load_and_process_image: {end_time - start_time:.4f} seconds")

        return img_stained, img_not_stained

    def read_band_images(self, band_files_dir):
        img_files = sorted(os.listdir(band_files_dir))
        image_bank = []
        for img_file in img_files:
            img_path = os.path.join(band_files_dir, img_file)
            img_gray = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_gray is None:
                raise ValueError(f"Failed to read grayscale image: {img_path}")
            image_bank.append(img_gray[:784] / 255.0)
        return image_bank

    def extract_patches(self, image):
        patches = []
        h, w = image.shape[:2]
        print("extract: ", h,w )
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                patch = image[i:i + self.patch_size, j:j + self.patch_size]
                #print("--------", patch.shape)
                patches.append(patch)
        # print("num of patches: ", len(patches)) #loads three patches
        #print(len(patches))
        return patches

def custom_collate_fn(batch):
    stained_patches = torch.cat([item[0] for item in batch], dim=0)
    not_stained_patches = torch.cat([item[1] for item in batch], dim=0)
    return stained_patches, not_stained_patches


def main(rank, world_size):
        
        dir1 = "/home/hkhz/remote_mnt/patches/stained_patches"
        dir2 = "/home/hkhz/remote_mnt/patches/not_stained_patches"
        dataset = PatchDataset(dir1, dir2, patch_size=256, stride=256)
        world_size = torch.cuda.device_count()
        #mp.spawn(main, args = (world_size,dataset,), nprocs=world_size, join=True)
    
        # Directly test dataset iteration without DataLoader
        #for i in range(len(dataset)):
        #   band_patches, rgb_patches = dataset[i]
        #  print(f"Item {i}: band_patches shape: {band_patches.shape}, rgb_patches shape: {rgb_patches.shape}")

        # Proceed with DataLoader if dataset iteration works
        start_tracing()
        print("in main")
        training_set, validation_set = train_test_split(dataset, test_size=0.2, random_state=42)
        print(len(dataset))
        print(len(training_set), len(validation_set))
        train_sampler = DistributedSampler(training_set, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(
            training_set,
            batch_size=2,
            shuffle=False,
            sampler = train_sampler,
            num_workers=0,
            pin_memory=True,
            #persistent_workers=True,
            collate_fn=custom_collate_fn,
            
        )
        validation_sampler = DistributedSampler(validation_set, num_replicas=world_size,rank=rank)
        validation_loader = DataLoader(
            validation_set,
            batch_size=2,
            shuffle=False,
            sampler =validation_sampler,
            num_workers=0,
            pin_memory=True,
            #persistent_workers=True,
            collate_fn=custom_collate_fn
        )


        train_loader = iter(train_loader)
        #print(next(dataloader))
        valid_loader = iter(validation_loader)

        
        def run_validation(vc, cur_iter, iterator_valid_bl, G_test_step, R_test_step, use_tqdm = True):
            valid_G_total_loss_list, valid_G_l1_loss_list, valid_G_ssim_list, valid_G_psnr_list, valid_G_ncc_list, \
            valid_D_real_loss_list, valid_D_fake_loss_list, valid_R_total_loss_list = [], [], [], [], [], [], [], []
            valid_idx_iterator = tqdm(range(vc.q_limit)) if use_tqdm else range(vc.q_limit)
            for i in valid_idx_iterator:
                if not use_tqdm:
                    print("Running Validation: {}/{}".format(i+1, vc.q_limit), end='\r')
                    if i == vc.q_limit - 1:
                        print()
                valid_y, valid_x = next(iterator_valid_bl)    
                valid_G_total_loss, valid_G_dis_loss, valid_G_l1_loss, valid_G_ssim, valid_G_psnr, valid_G_ncc = G_test_step(
                    valid_x, valid_y, 3)
                valid_D_total_loss, valid_D_real_loss, valid_D_fake_loss, valid_G_output = D_test_step(valid_x, valid_y)
                valid_R_total_loss, _, valid_R_output = R_test_step(valid_x, valid_y)
                #if i < 5:
                #   for j in range(vc.batch_size):
                #      if tc.label_channels == 1:
                #         plt.imsave(tc.model_path + f'/output/iter={cur_iter}_sample={i * vc.batch_size + j}_outputG.jpg',
                    #                valid_G_output[j, :, :, 0], cmap='gray')
                    #       plt.imsave(tc.model_path + f'/output/iter={cur_iter}_sample={i * vc.batch_size + j}_outputR.jpg',
                    #              valid_R_output[0][j, :, :, 0], cmap='gray')
                    #     plt.imsave(tc.model_path + f'/output/iter={cur_iter}_sample={i * vc.batch_size + j}_target.jpg',
                        #            valid_y[j, :, :, 0], cmap='gray')
                        #else:
                        #   plt.imsave(tc.model_path + f'/output/iter={cur_iter}_sample={i * vc.batch_size + j}_outputG.jpg',
                        #          np.clip(valid_G_output[j, :, :, :].numpy(), 0, 1))
                        #  plt.imsave(tc.model_path + f'/output/iter={cur_iter}_sample={i * vc.batch_size + j}_outputR.jpg',
                        #         np.clip(valid_R_output[0][j, :, :, :].numpy(), 0, 1))
                        # plt.imsave(tc.model_path + f'/output/iter={cur_iter}_sample={i * vc.batch_size + j}_target.jpg',
                            #       valid_y[j, :, :, :].numpy())
                            #plt.imsave(tc.model_path + f'/output/iter={cur_iter}_sample={i * vc.batch_size + j}_input0.jpg',
                            #      -valid_x[j, :, :, 0].numpy(), cmap='gray')
                valid_G_total_loss_list.append(valid_G_total_loss)
                valid_G_l1_loss_list.append(valid_G_l1_loss)
                valid_G_ssim_list.append(valid_G_ssim)
                valid_G_psnr_list.append(valid_G_psnr)
                valid_G_ncc_list.append(valid_G_ncc.numpy().item())
                #if valid_G_ncc.numpy().item() > 1:
                #   plt.imsave(
                #      tc.model_path + f'/output/wrongncc{valid_G_ncc.numpy().item()}_iter={cur_iter}_sample={i * vc.batch_size + j}_output.jpg',
                #     np.clip(valid_G_output[j, :, :, :].numpy(), 0, 1))
                # plt.imsave(
                    #    tc.model_path + f'/output/wrongncc{valid_G_ncc.numpy().item()}_iter={cur_iter}_sample={i * vc.batch_size + j}_label.jpg',
                    #    valid_y[j, :, :, :].numpy())
                valid_D_real_loss_list.append(valid_D_real_loss)
                valid_D_fake_loss_list.append(valid_D_fake_loss)
                valid_R_total_loss_list.append(valid_R_total_loss)
                valid_G_total_loss_mean = np.mean(np.array(valid_G_total_loss_list))
            valid_G_l1_loss_mean = np.mean(np.array(valid_G_l1_loss_list))
            valid_G_ssim_mean = np.mean(np.array(valid_G_ssim_list))
            valid_G_psnr_mean = np.mean(np.array(valid_G_psnr_list))
            vaild_G_ncc_mean = np.mean(np.array(valid_G_ncc_list))
            valid_G_ncc_std = np.std(np.array(valid_G_ncc_list))
            valid_D_real_loss_mean = np.mean(np.array(valid_D_real_loss_list))
            valid_D_fake_loss_mean = np.mean(np.array(valid_D_fake_loss_list))
            valid_R_total_loss_mean = np.mean(np.array(valid_R_total_loss_list))
            return valid_G_total_loss_mean, valid_G_l1_loss_mean, valid_G_ssim_mean, valid_G_psnr_mean, \
            vaild_G_ncc_mean, valid_G_ncc_std, valid_D_real_loss_mean, valid_D_fake_loss_mean, valid_R_total_loss_mean

        def init_parameters():
            tc, vc = ConfigObj(), ConfigObj()

            tc.model_path = '/home/hkhz/emil/Project_ai/voxelmorph_reg/models' # set the path to save model
            tc.prev_checkpoint_path = None
            tc.save_every_epoch = True

            # pretrained checkpoints to start from
            tc.G_warmstart_checkpoint = None
            tc.D_warmstart_checkpoint = None
            tc.R_warmstart_checkpoint = None
            assert not (tc.prev_checkpoint_path
                        and (tc.G_warmstart_checkpoint or tc.D_warmstart_checkpoint or tc.R_warmstart_checkpoint))

            tc.image_path = 'L:/Pneumonia_Dataset/Second_reg/Training/target/*.mat' # path for training data 
            vc.image_path = 'J:/Pneumonia_Dataset/Second_reg/Validation/target/*.mat' # path for validation data

            def convert_inp_path_from_target(inp_path: str):
                return inp_path.replace('target', 'input')

            tc.convert_inp_path_from_target = convert_inp_path_from_target
            vc.convert_inp_path_from_target = convert_inp_path_from_target

            tc.is_mat, vc.is_mat = True, True  # True for .mat, False for .npy
            tc.data_inpnorm, vc.data_inpnorm = 'norm_by_mean_std', 'norm_by_mean_std'
            tc.channel_start_index, vc.channel_start_index = 0, 0
            tc.channel_end_index, vc.channel_end_index = 2, 2  # exclusive

            # network and loss params
            tc.is_training, vc.is_training = True, False
            tc.image_size, vc.image_size = 256, 256
            tc.num_slices, vc.num_slices = 2, 2
            tc.label_channels, vc.label_channels = 3, 3
            assert tc.channel_end_index - tc.channel_start_index == tc.num_slices
            assert vc.channel_end_index - vc.channel_start_index == vc.num_slices
            tc.n_channels, vc.n_channels = 32, 32
            tc.lamda = 50.0  # adv loss

            tc.nf_enc, vc.nf_enc = [8, 16, 16, 32, 32], [8, 16, 16, 32, 32]  # for aligner
            tc.nf_dec, vc.nf_dec = [32, 32, 32, 32, 32, 16, 16], [32, 32, 32, 32, 32, 16, 16]  # for aligner
            tc.R_loss_type = 'ncc'
            tc.lambda_r_tv = 1.0  # .1    # tv of predicted flow
            tc.gauss_kernal_size = 80
            tc.dvf_clipping = True  # clip DVF to [mu-sigma*dvf_clipping_nsigma, mu+sigma*dvf_clipping_nsigma]
            tc.dvf_clipping_nsigma = 3
            tc.dvf_thresholding = True  # clip DVF to [-dvf_thresholding_distance, dvf_thresholding_distance]
            tc.dvf_thresholding_distance = 30

            # training params
            tc.batch_size, vc.batch_size = 4, 4
            tc.n_shuffle_epoch, vc.n_shuffle_epoch = 500, 5000  # for the batchloader
            tc.initial_alternate_steps = 6000  # train G/D for initial_alternate_steps steps before switching to R for the same # of steps
            tc.valid_steps = 5  # perform validation when D_steps % valid_steps == 0 or at the end of a loop of (train G/D, train R)
            tc.n_threads, vc.n_threads = 2, 2
            tc.q_limit, vc.q_limit = 100, 300
            tc.N_epoch = 150  # number of loops

            tc.tol = 0  # current early stopping patience
            tc.max_tol = 2  # the max-allowed early stopping patience
            tc.min_del = 0  # the lowest acceptable loss value reduction

            # case filtering
            tc.case_filtering = False
            tc.case_filtering_metric = 'ncc'  # 'ncc'
            # divide each patch into case_filtering_x_subdivision patches alone the x dimension for filtering (1 = no division)
            tc.case_filtering_x_subdivision = 1
            tc.case_filtering_y_subdivision = 1
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
            tc.epoch_begin = 79
            # this overrides tc.epoch_  begin the training schedule; tc.epoch_begin is required for logging
            # set it to None when not used
            tc.iter_begin =  None

            return tc, vc
        def default_unet_features():
            nb_features = [
                [16, 32, 64, 128],             # encoder
                [128, 128, 128, 64, 32, 16, 16]  # decoder
            ]
            return nb_features
        tc,vc = init_parameters()
        filters = [64,128,256, 512]
        ddp_setup(rank, world_size)
        model_G = att_unet_2d(input_size=(3, 256, 256), filter_num=filters, n_labels=3, rank=rank).to(rank)
        model_G = DDP(model_G, device_ids=[rank])

        model_D = discriminator_2d(input_size=(3, 256, 256), filter_num=filters, stack_num_down=2, activation='ReLU', batch_norm=False, pool=False, backbone=None).to(rank)
        model_D = DDP(model_D, device_ids=[rank])

        model_R = VxmDense(
            inshape=(256, 256),  # Adjust the inshape to match padded image size
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
        model_R = DDP(model_R = VxmDense(
            inshape=(256, 256),  # Adjust the inshape to match padded image size
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
        ).to([rank]), device_ids= [rank])
        G_optimizer = torch.optim.Adam(model_G.parameters(),lr=1e-4)
        D_optimizer = torch.optim.Adam(model_D.parameters(),lr=1e-5)
        R_optimizer = torch.optim.Adam(model_R.parameters(),lr=1e-4)

        def G_train_step(input_image, target, epoch):
            input_image = input_image.to(rank)
            target = target.to(rank)
            # ensure shuffling in multiprocessing
            train_sampler.set_epoch(epoch)
            model_G.train()
            model_D.eval()
            model_R.eval()
            G_optimizer.zero_grad()
            
            G_outputs = model_G(input_image)
            print("shape G: ",G_outputs.shape)
            print("shape target: ", target.shape)
            with torch.no_grad():
                D_fake_output = model_D(G_outputs)
                
            if epoch>0:
                start = time.perf_counter()
                time.sleep(1)
                with torch.no_grad(): 
                    # No gradients for target transformation
                    print("-----", target.shape, G_outputs.shape)
                    target_transformed, _ = model_R(target, G_outputs)
                end = time.perf_counter()
                elapsed = end - start
            else:
                target_transformed = target
            G_total_loss, G_dis_loss, G_l1_loss = loss_G(D_fake_output, G_outputs, target_transformed, tc, epoch)
            return G_total_loss, G_dis_loss, G_l1_loss, G_outputs, target_transformed
        def G_test_step(input_image, target, epoch):
            model_G.eval()
            model_D.eval()
            G_outputs = model_G(input_image)
            G_output_clipped_splitted = split_tensor(torch.clamp(G_outputs, 0, 1), tc.case_filtering_x_subdivision,
                                                        tc.case_filtering_y_subdivision)
            target_clipped_splitted = split_tensor(torch.clamp(target,0,1), tc.case_filtering_x_subdivision,
                                                        tc.case_filtering_y_subdivision)
            D_fake_output = model_D(G_outputs)
            G_total_loss, G_dis_loss, G_l1_loss = loss_G(D_fake_output, G_outputs, target, tc, epoch)
            G_ssim = compute_ssim(G_outputs, target)
            G_psnr = compute_psnr(G_outputs,target)
            G_ncc = torch.mean(NCC(win = 20, eps = 1e-3).ncc(target_clipped_splitted, G_output_clipped_splitted))
            return G_total_loss, G_dis_loss, G_l1_loss, G_ssim, G_psnr, G_ncc



        def D_train_step(input_image, target,epoch):
            train_sampler.set_epoch(epoch)
            model_D.train()
            model_G.eval()
            input_image, target = input_image.to(rank), target.to(rank)
            D_optimizer.zero_grad()
            with torch.no_grad():
                G_outputs = model_G(input_image)
            D_real_output = model_D(target)
            D_fake_output = model_D(G_outputs)
            D_total_loss, D_real_loss, D_fake_loss = loss_D(D_real_output, D_fake_output)
            
            D_total_loss.backward()
            D_optimizer.step()
            return D_total_loss, D_real_loss, D_fake_loss, G_outputs


        def D_test_step(input_image, target):
            model_G.eval()
            model_D.eval()
            G_outputs = model_G(input_image)
            D_real_output = model_D(target)
            D_fake_output = model_D(G_outputs)
            D_total_loss, D_real_loss, D_fake_loss = loss_D(D_real_output, D_fake_output)
            
            return D_total_loss, D_real_loss, D_fake_loss, G_outputs

        def R_train_step(input_image, target):
            train_sampler.set_epoch(epoch)
            model_R.train()
            model_G.eval()
            R_optimizer.zero_grad()
            with torch.no_grad():
                G_outputs = model_G(input_image)
            R_outputs = model_R(G_outputs, target)
            R_total_loss, R_berhu_loss = loss_R_no_gt(R_outputs, G_outputs, tc)
            R_total_loss.backward()
            R_optimizer.step()
            return R_total_loss, R_berhu_loss, R_outputs

        def R_test_step(input_image, target):
            model_G.eval()
            model_R.eval()
            G_outputs = model_G(input_image)
            R_outputs = model_R( G_outputs, target)
            R_total_loss, R_berhu_loss = loss_R_no_gt(R_outputs, G_outputs, tc)
            return R_total_loss, R_berhu_loss, R_outputs

        min_loss = 1e8
        max_psnr = 0
        epoch_begin = tc.epoch_begin
        iter_D_count = 0
        warmstart_first_epoch_elapsed_iters = None
        for i in range(epoch_begin):
            iter_D_count += max(int(tc.initial_alternate_steps * 0.9 ** i), 500)
        
        for epoch in range(epoch_begin, tc.N_epoch):
        
            print(f'Current iter_D_count: {iter_D_count}')

            train_G_total_loss_list, train_G_l1_loss_list, train_R_total_loss_list = [], [], []
            num_checkpoint_D = max(int(tc.initial_alternate_steps * 0.9 ** epoch), 500)
            num_checkpoint_R = max(int(tc.initial_alternate_steps * 0.9 ** epoch), 500)
            if epoch > 0 or (tc.G_warmstart_checkpoint is None and tc.prev_checkpoint_path is None):        
                print('training G & D ...')
                tc.epoch_filtering_ratio = []
                for i in tqdm(range(num_checkpoint_D)):
                    start = time.perf_counter()
                    time.sleep(1)
                    
                    NumGen = max(3, int(12 - iter_D_count //4000))
                    
                    for j in range(NumGen):
                        targets, input_images = next(train_loader)
                        input_images = input_images.to(rank)
                        targets = targets.to(rank)
                        train_G_total_loss, train_G_dis_loss, train_G_l1_loss, train_G_output, train_R_output = G_train_step(
                        input_images, targets, epoch)
                        train_G_total_loss_list.append(train_G_total_loss)
                        train_G_l1_loss_list.append(train_G_l1_loss)
                        
                    train_y, train_x = next(train_loader)
                    print("'fore D: ", train_x.shape, train_y.shape)
                    train_D_total_loss, train_D_real_loss, train_D_fake_loss, _ = D_train_step(train_x, train_y, epoch)
                    iter_D_count += 1

                    end = time.perf_counter() 
                    elapsed = end-start
                    print("D check:", num_checkpoint_D)
                    #######################################################################################################
                    # mid-round validation
                    #######################################################################################################
                    if iter_D_count % tc.valid_steps == 0 and i!= num_checkpoint_D-1:
                        valid_G_total_loss_mean, valid_G_l1_loss_mean, valid_G_ssim_mean, valid_G_psnr_mean, \
                            vaild_G_ncc_mean, valid_G_ncc_std, valid_D_real_loss_mean, valid_D_fake_loss_mean, \
                            valid_R_total_loss_mean = run_validation(vc, iter_D_count, valid_loader, G_test_step,
                                                                    D_test_step, R_test_step, use_tqdm=False) #TODO: change to validation loader
                        train_G_total_loss_mean = np.mean(np.array(train_G_total_loss_list))
                        train_G_l1_loss_mean = np.mean(np.array(train_G_l1_loss_list))
                        train_R_total_loss_mean = np.mean(np.array(train_R_total_loss_list))
                        print("Mid round Losses: ")
                        print("train_G_total_loss_mean: " , train_G_total_loss_mean)
                        print("train_G_l1_loss_mean: " , train_G_l1_loss_mean)
                        print("train_R_total_loss_mean: " , train_R_total_loss_mean)
                    
                        #train_G_total_loss_list, train_G_l1_loss_list, train_R_total_loss_list = [], [], []
                        #if min_loss - valid_G_l1_loss_mean > tc.min_del or valid_G_psnr_mean > max_psnr \
                        #        or tc.save_every_epoch:
                        #    # tol = 0  # refresh early stopping patience
                        #   torch.save(model_G.state_dict,tc.model_path + f'/model_G_iter={iter_D_count}.h5')
                        #  torch.save(model_D.state_dict,tc.model_path + f'/model_D_iter={iter_D_count}.h5')
                        torch.save(model_R.module.state_dict, tc.model_path + f'/model_R_iter={iter_D_count}.h5')

                        # if min_loss - valid_G_l1_loss_mean > tc.min_del:
                            #    print('Validation loss is improved from {} to {}'.format(min_loss, valid_G_l1_loss_mean))
                            #   min_loss = valid_G_l1_loss_mean  # update the loss record

                            #if valid_G_psnr_mean > max_psnr:
                            #  print('Validation PSNR is improved from {} to {}'.format(max_psnr, valid_G_psnr_mean))
                            #   max_psnr = valid_G_psnr_mean
                    #  torch.save(model_G.state_dict,tc.model_path + f'/model_G_latest.h5')
                    # torch.save(model_D.state_dict,tc.model_path + f'/model_D_latest.h5')
                    # torch.save(model_R.state_dict,tc.model_path + f'/model_R_latest.h5')
                        #np.save(tc.model_path + f'/optimizer_G_latest.npy', G_optimizer.get_weights())
                        #np.save(tc.model_path + f'/optimizer_D_latest.npy', D_optimizer.get_weights())
                        #np.save(tc.model_path + f'/optimizer_R_latest.npy', R_optimizer.get_weights())
            print('training R ...')
            for i in tqdm(range(num_checkpoint_R)):
                start = time.perf_counter()
                time.sleep(1)
                train_x, train_y = next(train_loader)
                train_R_total_loss, _, train_R_output = R_train_step(train_x, train_y, epoch)
                #wtr.check_stop()
                train_R_total_loss_list.append(train_R_total_loss)
                end = time.perf_counter() 
                elapsed = end-start
            ###############################################################################################################
            # round-end validation
            ###############################################################################################################
            valid_G_total_loss_mean, valid_G_l1_loss_mean, valid_G_ssim_mean, valid_G_psnr_mean, \
                vaild_G_ncc_mean, valid_G_ncc_std, valid_D_real_loss_mean, valid_D_fake_loss_mean, \
                valid_R_total_loss_mean = \
                run_validation(vc, iter_D_count, valid_loader, G_test_step, D_test_step, R_test_step)#TODO create a validation
            print("Losses: ")
            print("valid_G_total_loss_mean: " , valid_G_total_loss_mean)
            print("valid_G_l1_loss_mean: " , valid_G_l1_loss_mean)
            print("valid_G_ssim_mean: " , valid_G_ssim_mean)
            print("valid_G_psnr_mean: " , valid_G_psnr_mean)
            print("vaild_G_ncc_mean: " , vaild_G_ncc_mean)
            print("valid_G_ncc_std: " , valid_G_ncc_std)
            print("valid_D_real_loss_mean: " , valid_D_real_loss_mean)
            print("valid_D_fake_loss_mean: " , valid_D_fake_loss_mean)
            print("valid_R_total_loss_mean: " , valid_R_total_loss_mean)
        cleanup()

        stop_tracing()
        print_memory_usage_by_function()




if __name__ == "__main__":
    # Assuming you have a list of pairs of img_bank and rgb_image
    # Each pair consists of (img_bank, rgb_image)
    pairs = [
        ([np.random.rand(256, 256, 3) for _ in range(300)], np.random.rand(256, 256, 3)),
        ([np.random.rand(256, 256, 3) for _ in range(300)], np.random.rand(256, 256, 3)),
        ([np.random.rand(256, 256, 3) for _ in range(300)], np.random.rand(256, 256, 3)),
        ([np.random.rand(256, 256, 3) for _ in range(300)], np.random.rand(256, 256, 3)),
        ([np.random.rand(256, 256, 3) for _ in range(300)], np.random.rand(256, 256, 3)),
        ([np.random.rand(256, 256, 3) for _ in range(300)], np.random.rand(256, 256, 3))

        # Add more pairs here as needed
        
    ]
   
    world_size = torch.cuda.device_count()
    mp.spawn(main, args = (world_size,), nprocs=world_size, join=True)
   
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import concurrent.futures
import time
from sklearn.model_selection import train_test_split

class PatchDataset(Dataset):
    def __init__(self, dir1, dir2, patch_size=256, stride=256, transform=None):
        self.dir1 = dir1
        self.dir2 = dir2
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.image_paths = self.get_image_paths()

    def get_image_paths(self):
        image_paths = []
        filenames = os.listdir(self.dir1)
        for filename in filenames:
            path_stained = os.path.join(self.dir1, filename)
            path_not_stained = os.path.join(self.dir2, 'not_'+filename)
            #band_files_dir = os.path.join(self.dir1, dir, f'{i:03d}')
            if os.path.exists(path_stained) and os.path.exists(path_not_stained):
                image_paths.append((path_stained, path_not_stained))
            else:
                logging.warning(f"Missing path_rgb: {path_stained} or band_files_dir: {path_not_stained}")
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        stained_image_p, not_stained_image_p = self.image_paths[idx]
        try:
            img_stained, img_not_stained = self.load_and_process_image(stained_image_p, not_stained_image_p)
            #print("shape: ", img_bank[0].shape)
        except Exception as e:
            logging.error(f"Error loading image {stained_image_p} or {not_stained_image_p}: {e}")
            raise e
        
        logging.info(f"Successfully loaded image {stained_image_p} and {not_stained_image_p}")

        stained_patches = self.extract_patches(img_stained) # each 256x256 from original 1024x1024 image
        not_stained_patches = self.extract_patches(img_not_stained) #same with not stained images

        patches = []
        for i in range(len(stained_patches)):
            stained_patch = stained_patches[i]
            #print("**", stained_patch.shape)
            not_stained_patch = not_stained_patches[i]
            #print("++", not_stained_patch.shape)
            patches.append((stained_patch, not_stained_patch))
        #print(patches[0][0])
        if self.transform:
            # Transform each band rgb
            patches = [(self.transform(pair[0]), self.transform(pair[1])) for pair in patches]
        #print("shape patch: ", patches[0][0].shape, patches[0][1].shape)
        stained_patches = torch.stack([torch.from_numpy(stained).permute(2, 0, 1).float() for stained, _ in patches])
        not_stained_patches = torch.stack([torch.from_numpy(not_stained).permute(2, 0, 1).float() for _, not_stained in patches])

        # Extract only one channel of the loaded hyperspectral
        #band_patches = torch.stack([torch.stack([torch.from_numpy(band[:, :, 0]).float() for band in bands], dim=0) for _, bands in patches])
        #rint("band shape ", band_patches.shape)
        #return rgb_patches, band_patches
        return patches

    def load_and_process_image(self, path_stained, path_not_stained):
        start_time = time.time()

        img_stained = cv2.imread(path_stained, cv2.IMREAD_COLOR)
        img_not_stained = cv2.imread(path_not_stained, cv2.IMREAD_COLOR)
        #print("Image shapes: ", img_stained.shape, img_not_stained.shape)
        if img_stained is None or img_not_stained is None:
            raise ValueError(f"Failed to read RGB image: {path_stained} or {path_not_stained}")
        #img_rgb = img_rgb[img_rgb.shape[0] - 784 - 29: -29] / 255.0

        #image_bank = self.read_band_images(band_files_dir)

        end_time = time.time()
        logging.info(f"Time taken for load_and_process_image: {end_time - start_time:.4f} seconds")

        return img_stained, img_not_stained

    def read_band_images(self, band_files_dir):
        img_files = sorted(os.listdir(band_files_dir))
        image_bank = []
        for img_file in img_files:
            img_path = os.path.join(band_files_dir, img_file)
            img_gray = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_gray is None:
                raise ValueError(f"Failed to read grayscale image: {img_path}")
            image_bank.append(img_gray[:784] / 255.0)
        return image_bank

    def extract_patches(self, image):
        patches = []
        h, w = image.shape[:2]
        print("extract: ", h,w )
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                patch = image[i:i + self.patch_size, j:j + self.patch_size]
                #print("--------", patch.shape)
                patches.append(patch)
        # print("num of patches: ", len(patches)) #loads three patches
        #print(len(patches))
        return patches

def custom_collate_fn(batch):
    rgb_patches = torch.cat([item[0] for item in batch], dim=0)
    band_patches = torch.cat([item[1] for item in batch], dim=0)
    return rgb_patches, band_patches


if __name__ == "__main__":
    dir1 = "/home/hkhz/remote_mnt/patches/stained_patches"
    dir2 = "/home/hkhz/remote_mnt/patches/not_stained_patches"
    print(dir2.split('_')[-1])
    dataset = PatchDataset(dir1, dir2, patch_size=256, stride=256)
    for i in range(len(dataset)):
        patch = dataset[i]
    """
"""
    # Directly test dataset iteration without DataLoader
    #for i in range(len(dataset)):
     #   band_patches, rgb_patches = dataset[i]
      #  print(f"Item {i}: band_patches shape: {band_patches.shape}, rgb_patches shape: {rgb_patches.shape}")

    # Proceed with DataLoader if dataset iteration works
    training_set, validation_set = train_test_split(dataset, test_size=0.2, random_state=42)
    print(len(dataset))
    print(len(training_set), len(validation_set))
    
    train_loader = DataLoader(
        training_set,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )


    train_loader = iter(train_loader)
    #print(next(dataloader))
    valid_loader = iter(validation_loader)
"""