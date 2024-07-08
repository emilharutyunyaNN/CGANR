from GAN_torch.att_unet_2d import att_unet_2d
from GAN_torch.discriminator_2d import discriminator_2d
from GAN_torch.losses_for_gan import *
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
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
import gc
import sys
from torch.cuda.amp import GradScaler, autocast
sys.path.append("./voxelmorph_reg/aligners")
from aligners.aligner_affine import AlignerUNetCVPR2018V4
from aligners.stn_affine import affine_transform
scaler = GradScaler()
#from concurrent.futures import ThreadPoolExecutor, as_completed
#import multiprocessing
from sklearn.model_selection import train_test_split
#multiprocessing.set_start_method('spawn', force=True)
# Define device
import matplotlib.pyplot as plt
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
logging.basicConfig(filename = 'gan_r_losses.log',format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
logger = logging.getLogger()
logger.info("------------------------------NEW RUN------------------------------")

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTERPORT"] = 12355
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    destroy_process_group() 



class PatchDataset(Dataset):
    def __init__(self, dir1, dir2, indices = None, patch_size=256, stride=256, transform=None, num_workers = 8):
        self.dir1 = dir1
        self.dir2 = dir2
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.num_workers = num_workers
        self.image_paths = self.get_image_paths()
        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]

    def get_image_paths(self):
        image_paths = []
        filenames = os.listdir(self.dir1)

        def process_file(filename):
            path_stained = os.path.join(self.dir1, filename)
            path_not_stained = os.path.join(self.dir2, 'not_' + filename)
            if os.path.exists(path_stained) and os.path.exists(path_not_stained):
                #image_paths.append((path_stained, path_not_stained))
                #print("Stained: ", path_stained, "   Not stained: ", path_not_stained)
                return path_stained, path_not_stained
            else:
                logging.warning(f"Missing path_rgb: {path_stained} or path_not_stained: {path_not_stained}")
                return None
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(process_file, filename) for filename in filenames]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    image_paths.append(result)
            
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        stained_image_p, not_stained_image_p = self.image_paths[idx]
        try:
            img_stained, img_not_stained = self.load_and_process_image(stained_image_p, not_stained_image_p)
        except Exception as e:
            logging.error(f"Error loading image {stained_image_p} or {not_stained_image_p}: {e}")
            raise e

        stained_patches = self.extract_patches(img_stained)
        not_stained_patches = self.extract_patches(img_not_stained)

        if self.transform:
            stained_patches = [self.transform(patch) for patch in stained_patches]
            not_stained_patches = [self.transform(patch) for patch in not_stained_patches]

        stained_patches = torch.stack([torch.from_numpy(patch).permute(2, 0, 1).float() for patch in stained_patches])
        not_stained_patches = torch.stack([torch.from_numpy(patch).permute(2, 0, 1).float() for patch in not_stained_patches])

        return stained_patches, not_stained_patches
    def load_and_process_image(self, path_stained, path_not_stained):
        img_stained = cv2.imread(path_stained, cv2.IMREAD_COLOR)
        img_stained = (img_stained - np.mean(img_stained)) / (np.std(img_stained) + 1e-5) 
        img_not_stained = cv2.imread(path_not_stained, cv2.IMREAD_COLOR)
        img_not_stained = (img_not_stained - np.mean(img_not_stained)) / (np.std(img_not_stained) + 1e-5) 
        if img_stained is None or img_not_stained is None:
            raise ValueError(f"Failed to read RGB image: {path_stained} or {path_not_stained}")
        return img_stained, img_not_stained

    def extract_patches(self, image):
        patches = []
        h, w = image.shape[:2]
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                patch = image[i:i + self.patch_size, j:j + self.patch_size]
                patches.append(patch)
        return patches

def custom_collate_fn(batch):
    stained_patches = torch.cat([item[0] for item in batch], dim=0)
    not_stained_patches = torch.cat([item[1] for item in batch], dim=0)
    return stained_patches, not_stained_patches

#def main(rank, world_size):
        
dir1 = "/home/hkhz/final/stained_patches"
dir2 = "/home/hkhz/final/not_stained_patches"
dataset = PatchDataset(dir1, dir2, patch_size=256, stride=256)
#world_size = torch.cuda.device_count()
#mp.spawn(main, args = (world_size,dataset,), nprocs=world_size, join=True)

# Directly test dataset iteration without DataLoader
#for i in range(len(dataset)):
#   band_patches, rgb_patches = dataset[i]
#  print(f"Item {i}: band_patches shape: {band_patches.shape}, rgb_patches shape: {rgb_patches.shape}")

# Proceed with DataLoader if dataset iteration works
start_tracing()
print("in main")
length_dataset = len(dataset)
train_length = int(0.8*length_dataset)
indices = list(range(length_dataset))
training_set_ind, validation_set_ind = indices[:train_length], indices[train_length:]
print(len(dataset))
print(len(training_set_ind), len(validation_set_ind))
#train_sampler = DistributedSampler(training_set, num_replicas=world_size, rank=rank)
training_set = PatchDataset(dir1,dir2, indices=training_set_ind)
validation_set = PatchDataset(dir1,dir2, indices=validation_set_ind)
train_loader = DataLoader(
    training_set,
    batch_size=1,
    shuffle=True,
    #sampler = train_sampler,
    num_workers=0,
    pin_memory=True,
    #persistent_workers=True,
    collate_fn=custom_collate_fn,
    
)
#validation_sampler = DistributedSampler(validation_set, num_replicas=world_size,rank=rank)
validation_loader = DataLoader(
    validation_set,
    batch_size=1,
    shuffle=False,
    #sampler =validation_sampler,
    num_workers=0,
    pin_memory=True,
    #persistent_workers=True,
    collate_fn=custom_collate_fn
)


train_loader = iter(train_loader)
#print(next(dataloader))
valid_loader = iter(validation_loader)


def run_validation(vc, cur_iter, iterator_valid_bl, D_test_step,G_test_step, R_test_step, use_tqdm = True):
    valid_G_total_loss_list, valid_G_l1_loss_list, valid_G_ssim_list, valid_G_psnr_list, valid_G_ncc_list, \
    valid_D_real_loss_list, valid_D_fake_loss_list, valid_R_total_loss_list, valid_R_affine_loss_list = [], [], [], [], [], [], [], [], []
    valid_idx_iterator = tqdm(range(vc.q_limit)) if use_tqdm else range(vc.q_limit)
    #print(valid_idx_iterator)
    
    for i in valid_idx_iterator:
        if not use_tqdm:
            print("Running Validation: {}/{}".format(i+1, vc.q_limit), end='\r')
            if i == vc.q_limit - 1:
                print()
        valid_y, valid_x = next(iterator_valid_bl)    
        valid_G_total_loss, valid_G_dis_loss, valid_G_l1_loss, valid_G_ssim, valid_G_psnr, valid_G_ncc = G_test_step(
            valid_x, valid_y, 3)
        valid_D_total_loss, valid_D_real_loss, valid_D_fake_loss, valid_G_output = D_test_step(valid_x, valid_y)
        valid_R_affine_loss, valid_R_total_loss, _, valid_R_output = R_test_step(valid_x, valid_y)
        print("R test overall: ", valid_R_total_loss)
        print("R test affine: ", valid_R_affine_loss)
        #print("done")
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
        valid_G_ncc_list.append(valid_G_ncc.detach().cpu().numpy().item())
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
        valid_R_affine_loss_list.append(valid_R_affine_loss)
    print(valid_R_total_loss_list)
    valid_G_total_loss_mean = np.mean(np.array([t.cpu().numpy() for t in valid_G_total_loss_list]))
    valid_G_l1_loss_mean = np.mean(np.array([t.cpu().numpy() for t in valid_G_l1_loss_list]))
    valid_G_ssim_mean = np.mean(np.array([t.cpu().numpy() for t in valid_G_ssim_list]))
    valid_G_psnr_mean = np.mean(np.array([t.cpu().numpy() for t in valid_G_psnr_list]))
    valid_G_ncc_mean = np.mean(np.array(valid_G_ncc_list))
    valid_G_ncc_std = np.std(np.array(valid_G_ncc_list))
    valid_D_real_loss_mean = np.mean(np.array([t.cpu().numpy() for t in valid_D_real_loss_list]))
    valid_D_fake_loss_mean = np.mean(np.array([t.cpu().numpy() for t in valid_D_fake_loss_list]))
    valid_R_total_loss_mean = np.nanmean(np.array([t.cpu().numpy() for t in valid_R_total_loss_list]))
    valid_R_affine_loss_mean = np.nanmean(np.array([t.cpu().numpy() for t in valid_R_affine_loss_list]))
    return valid_G_total_loss_mean, valid_G_l1_loss_mean, valid_G_ssim_mean, valid_G_psnr_mean, valid_G_ncc_mean, valid_G_ncc_std, valid_D_real_loss_mean, valid_D_fake_loss_mean, valid_R_total_loss_mean, valid_R_affine_loss_mean

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
    tc.valid_steps = 100  # perform validation when D_steps % valid_steps == 0 or at the end of a loop of (train G/D, train R)
    tc.n_threads, vc.n_threads = 2, 2
    tc.q_limit, vc.q_limit = 30, 10
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
#ddp_setup(rank, world_size)
model_G = att_unet_2d(input_size=(3, 256, 256), filter_num=filters, n_labels=3, rank=1,stack_num_down=3, stack_num_up=3, activation='LeakyReLU',
                          atten_activation='ReLU', attention='add',
                          output_activation=None, batch_norm=True, pool='ave', unpool='bilinear')
#model_G = DDP(model_G, device_ids=[rank])
model_G.to(device)

model_D = discriminator_2d(input_size=(3, 256, 256), filter_num=filters, rank = 1, stack_num_down=1, activation='LeakyReLU',
                               batch_norm=False, pool=False)
#model_D = DDP(model_D, device_ids=[rank])
model_D.to(device)
model_R_affine = AlignerUNetCVPR2018V4(
    vol_size=(256,256),
    enc_nf=[64,128,256],
    dec_nf=[256,128,64],
    src_feats=3,
    tgt_feats=3
)
model_R_affine.to(device)




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
)
model_R.to(device)

G_optimizer = torch.optim.Adam(model_G.parameters(),lr=1e-4)
D_optimizer = torch.optim.Adam(model_D.parameters(),lr=1e-5)
R_optimizer = torch.optim.Adam(model_R.parameters(),lr=1e-4)
R_affine_optimizer = torch.optim.Adam(model_R_affine.parameters(),lr=1e-4)

def G_train_step(input_image, target, epoch):
    #print("fore input: ", torch.cuda.memory_allocated())
    input_image = input_image.float().to(device)
    target = target.float().to(device)
   # print("after input : ", torch.cuda.memory_allocated())
    #print("input: ", input_image.shape)
    #print("target: ", target.shape)
    # ensure shuffling in multiprocessing
    #train_sampler.set_epoch(epoch)
    model_G.train()
    model_D.eval()
    model_R.eval()
    model_R_affine.eval()
    G_optimizer.zero_grad()
    with autocast():
        G_outputs = model_G(input_image).float()
    #print("after model G : ", torch.cuda.memory_allocated())
    #print("shape G: ",G_outputs.shape)
    #print("shape target: ", target.shape)
       # print(type(target), type(G_outputs))
        with torch.no_grad():
            D_fake_output = model_D(G_outputs)
            
        if epoch>0:
            start = time.perf_counter()
            time.sleep(1)
            with torch.no_grad(): 
                # No gradients for target transformation
            # print("-----", target.shape, G_outputs.shape)
                #print(model_R_affine(target, G_outputs)[0].shape)
                R_affine_output = model_R_affine(target, G_outputs)[0].to(torch.float32)
                assert not torch.isnan(R_affine_output).any(), "NaN detected in R_affine_outputs during training"
                target_transformed, _ = model_R(R_affine_output, G_outputs)
                #target_transformed, _ = model_R(target, G_outputs)
            end = time.perf_counter()
            elapsed = end - start
        else:
            target_transformed = target
   # print("fore input del : ", torch.cuda.memory_allocated())
    
    
    #print("after input del : ", torch.cuda.memory_allocated())
    
        G_total_loss, G_dis_loss, G_l1_loss = loss_G(D_fake_output, G_outputs, target_transformed, tc, epoch)
    scaler.scale(G_total_loss).backward()
    scaler.step(G_optimizer)
    scaler.update()
    del input_image 
    del target
    gc.collect()
    torch.cuda.empty_cache()
    return G_total_loss, G_dis_loss, G_l1_loss
    return G_total_loss, G_dis_loss, G_l1_loss, G_outputs, target_transformed
def G_test_step(input_image, target, epoch):
    input_image, target = input_image.to(device), target.to(device)

    model_G.eval()
    model_D.eval()
    with torch.no_grad():
        G_outputs = model_G(input_image)
    G_output_clipped_splitted = split_tensor(torch.clamp(G_outputs, 0, 1), tc.case_filtering_x_subdivision,
                                                tc.case_filtering_y_subdivision)
    target_clipped_splitted = split_tensor(torch.clamp(target,0,1), tc.case_filtering_x_subdivision,
                                                tc.case_filtering_y_subdivision)
    with torch.no_grad():
        D_fake_output = model_D(G_outputs)
    G_total_loss, G_dis_loss, G_l1_loss = loss_G(D_fake_output, G_outputs, target, tc, epoch)
    G_ssim = compute_ssim(G_outputs, target)
    G_psnr = compute_psnr(G_outputs,target)
   # print("+++", target_clipped_splitted.shape, G_output_clipped_splitted.shape)
    G_ncc = torch.mean(NCC(win = 20, eps = 1e-3).ncc(target_clipped_splitted, G_output_clipped_splitted))
    return G_total_loss, G_dis_loss, G_l1_loss, G_ssim, G_psnr, G_ncc



def D_train_step(input_image, target,epoch):
   # train_sampler.set_epoch(epoch)
    model_D.train()
    model_G.eval()
    input_image, target = input_image.to(device), target.to(device)
    D_optimizer.zero_grad()
    with torch.no_grad():
        G_outputs = model_G(input_image)
    with autocast():
        D_real_output = model_D(target)
        D_fake_output = model_D(G_outputs)
        D_total_loss, D_real_loss, D_fake_loss = loss_D(D_real_output, D_fake_output)
    
    scaler.scale(D_total_loss).backward()
    scaler.step(D_optimizer)
    scaler.update()
    torch.cuda.empty_cache()
    return D_total_loss, D_real_loss, D_fake_loss, G_outputs


def D_test_step(input_image, target):
    model_G.eval()
    model_D.eval()
    input_image, target = input_image.to(device), target.to(device)
    with torch.no_grad():
        G_outputs = model_G(input_image)
        D_real_output = model_D(target)
        D_fake_output = model_D(G_outputs)
    D_total_loss, D_real_loss, D_fake_loss = loss_D(D_real_output, D_fake_output)
    
    return D_total_loss, D_real_loss, D_fake_loss, G_outputs

def R_train_step(input_image, target):
   # train_sampler.set_epoch(epoch)
    input_image, target = input_image.to(device), target.to(device)
    model_R.train()
    model_R_affine.eval()
    model_G.eval()
    R_optimizer.zero_grad()
    #R_affine_optimizer.zero_grad()
    with torch.no_grad():
        G_outputs = model_G(input_image)
    with autocast():
        #R_outputs_affine = model_R_affine(target, G_outputs)[0]
        with torch.no_grad(): 
            R_affine_outputs, _ = model_R_affine(target, G_outputs)
            R_affine_outputs = R_affine_outputs.float()
        assert not torch.isnan(R_affine_outputs).any(), "NaN detected in R_affine_outputs during training"
        #R_outputs = model_R(R_outputs_affine, G_outputs)
        R_outputs = model_R(R_affine_outputs, G_outputs)
        R_total_loss, R_berhu_loss = loss_R_no_gt(R_outputs, G_outputs, tc)

    scaler.scale(R_total_loss).backward()
    scaler.step(R_optimizer)
   # scaler.step(R_affine_optimizer)
    scaler.update()
    torch.cuda.empty_cache()
    return R_total_loss, R_berhu_loss, R_outputs

def R_test_step(input_image, target):
    model_G.eval()
    model_R.eval()
    model_R_affine.eval()
    input_image, target = input_image.to(device), target.to(device)
    with torch.no_grad():
        G_outputs = model_G(input_image)
        R_affine_outputs, affine = model_R_affine(target, G_outputs)
        R_affine_outputs = R_affine_outputs.float()
        assert not torch.isnan(R_affine_outputs).any(), "NaN detected in R_affine_outputs during training"
        R_outputs= model_R(R_affine_outputs, G_outputs)
        #R_outputs = model_R(target, G_outputs)
        #print("dvf on affine:", R_outputs[0])
    R_affine_loss = affine_transformation_loss(R_outputs[0], G_outputs, affine)
    R_total_loss, R_berhu_loss = loss_R_no_gt(R_outputs, G_outputs, tc)
    return R_affine_loss, R_total_loss, R_berhu_loss, R_outputs

def R_affine_train(input_image, target):
    input_image, target = input_image.to(device), target.to(device)
    model_R_affine.train()
    #model_R_affine.train()
    model_R.eval()
    model_G.eval()
    R_affine_optimizer.zero_grad()
    #R_affine_optimizer.zero_grad()
    with torch.no_grad():
        G_outputs = model_G(input_image)
    with autocast():
        #R_outputs_affine = model_R_affine(target, G_outputs)[0]
        #R_outputs = model_R(R_outputs_affine, G_outputs)
        R_outputs, affine = model_R_affine(target, G_outputs)
        with torch.no_grad(): 
            R_outputs, _ = model_R(R_outputs, G_outputs)
        R_affine_loss= affine_transformation_loss(R_outputs, G_outputs, affine)

    scaler.scale(R_affine_loss).backward()
    scaler.step(R_affine_optimizer)
   # scaler.step(R_affine_optimizer)
    scaler.update()
    torch.cuda.empty_cache()
    return R_affine_loss, R_outputs
    


min_loss = 1e8
max_psnr = 0
epoch_begin = tc.epoch_begin
iter_D_count = 0
warmstart_first_epoch_elapsed_iters = None
for i in range(epoch_begin):
    iter_D_count += max(int(tc.initial_alternate_steps * 0.9 ** i), 500)
print(iter_D_count)
#iter_D_count = 100
for epoch in range(epoch_begin, tc.N_epoch):
    print(f"Epoch {epoch} started training ...")

    print(f'Current iter_D_count: {iter_D_count}')

    train_G_total_loss_list, train_G_l1_loss_list, train_R_total_loss_list, train_R_affine_loss_list = [], [], [], []
    num_checkpoint_D = max(int(tc.initial_alternate_steps * 0.9 ** epoch), 500)
    num_checkpoint_R = max(int(tc.initial_alternate_steps * 0.9 ** epoch), 500)
    if epoch > 0 or (tc.G_warmstart_checkpoint is None and tc.prev_checkpoint_path is None):        
        print('training G & D ...')
        tc.epoch_filtering_ratio = []
        #num_checkpoint_D=120
        for i in tqdm(range(num_checkpoint_D)):
            start = time.perf_counter()
            time.sleep(1)
            
            NumGen = max(3, int(12 - iter_D_count //4000))
            
            for j in range(NumGen):
                targets, input_images = next(train_loader)
                input_images = input_images.to(device)
                targets = targets.to(device)
                #train_G_total_loss, train_G_dis_loss, train_G_l1_loss, train_G_output, train_R_output = G_train_step(
                #input_images, targets, epoch)
                train_G_total_loss, train_G_dis_loss, train_G_l1_loss = G_train_step(
                input_images, targets, epoch)
                train_G_total_loss_list.append(train_G_total_loss)
                train_G_l1_loss_list.append(train_G_l1_loss)
                
            #print("training fore: ", torch.cuda.memory_allocated())
            train_y, train_x = next(train_loader)
            #print("training after : ", torch.cuda.memory_allocated())
           # print("'fore D: ", train_x.shape, train_y.shape)
            train_D_total_loss, train_D_real_loss, train_D_fake_loss, _ = D_train_step(train_x, train_y, epoch)
            iter_D_count += 1
            #del train_D_total_loss, train_D_real_loss, train_D_fake_loss
          #  print("memory allocated : ", torch.cuda.memory_allocated())
            end = time.perf_counter() 
            elapsed = end-start
          #  print("D check:", num_checkpoint_D)
            torch.cuda.empty_cache()
            #######################################################################################################
            # mid-round validation
            #######################################################################################################
            if iter_D_count % tc.valid_steps == 0 and i!= num_checkpoint_D-1:
                valid_G_total_loss_mean, valid_G_l1_loss_mean, valid_G_ssim_mean, valid_G_psnr_mean, \
                    vaild_G_ncc_mean, valid_G_ncc_std, valid_D_real_loss_mean, valid_D_fake_loss_mean, \
                    valid_R_total_loss_mean, valid_R_affine_loss_mean= run_validation(vc, iter_D_count, valid_loader, D_test_step,
                                                            G_test_step, R_test_step, use_tqdm=False) #TODO: change to validation loader
                
                logger.info(f"------------------------MID-ROUND VALIDATION------------------------")
                print("Losses: ")
                print("valid_G_total_loss_mean: " , valid_G_total_loss_mean)
                logger.info(f"Validation G total loss: {valid_G_total_loss_mean}")
                print("valid_G_l1_loss_mean: " , valid_G_l1_loss_mean)
                logger.info(f"Validation G L1 loss: {valid_G_l1_loss_mean}")
                print("valid_G_ssim_mean: " , valid_G_ssim_mean)
                logger.info(f"Validation G SSIM loss: {valid_G_ssim_mean}")
                print("valid_G_psnr_mean: " , valid_G_psnr_mean)
                logger.info(f"Validation G PSNR loss: {valid_G_psnr_mean}")
                print("vaild_G_ncc_mean: " , vaild_G_ncc_mean)
                logger.info(f"Validation G NCC loss mean: {vaild_G_ncc_mean}")
                print("valid_G_ncc_std: " , valid_G_ncc_std)
                logger.info(f"Validation G NCC loss std: {valid_G_ncc_std}")
                print("valid_D_real_loss_mean: " , valid_D_real_loss_mean)
                logger.info(f"Validation D real loss: {valid_D_real_loss_mean}")
                print("valid_D_fake_loss_mean: " , valid_D_fake_loss_mean)
                logger.info(f"Validation D fake loss: {valid_D_fake_loss_mean}")
                print("valid_R_total_loss_mean: " , valid_R_total_loss_mean)
                logger.info(f"Validation R total loss: {valid_R_total_loss_mean}")
                print("valid_R_affine_loss_mean: " , valid_R_affine_loss_mean)
                logger.info(f"Validation R total loss: {valid_R_affine_loss_mean}")
                logger.info(f"------------------------MID-ROUND VALIDATION------------------------")
                
            
                #train_G_total_loss_list, train_G_l1_loss_list, train_R_total_loss_list = [], [], []
                #if min_loss - valid_G_l1_loss_mean > tc.min_del or valid_G_psnr_mean > max_psnr \
                #        or tc.save_every_epoch:
                #    # tol = 0  # refresh early stopping patience
                #   torch.save(model_G.state_dict,tc.model_path + f'/model_G_iter={iter_D_count}.h5')
                #  torch.save(model_D.state_dict,tc.model_path + f'/model_D_iter={iter_D_count}.h5')
                torch.save(model_R.state_dict, tc.model_path + f'/model_R_latest.h5')
                torch.save(model_R_affine.state_dict, tc.model_path + f'/model_R_affine_latest.h5')
                torch.save(model_G.state_dict, tc.model_path + f'/model_G_latest.h5')
                torch.save(model_D.state_dict, tc.model_path + f'/model_D_latest.h5')

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
    train_G_total_loss_mean = np.mean(np.array([ t.detach().cpu().numpy() for t in train_G_total_loss_list]))
    train_G_l1_loss_mean = np.mean(np.array([ t.detach().cpu().numpy() for t in train_G_l1_loss_list]))
    print("Training losses")
    logger.info("-------------------TRAINING LOSSES-------------------")
    print("train_G_total_loss_mean: " , train_G_total_loss_mean)
    logger.info(f"Total loss for G: {train_G_total_loss_mean}")
    print("train_G_l1_loss_mean: " , train_G_l1_loss_mean)
    logger.info(f"L1 loss for G: { train_G_l1_loss_mean}")
    print('training R ...')
    #num_checkpoint_R = 120
    for i in tqdm(range(num_checkpoint_R)):
        start = time.perf_counter()
        time.sleep(1)
        train_x, train_y = next(train_loader)
        train_R_total_loss, _, train_R_output = R_train_step(train_y, train_x)
        #wtr.check_stop()
        train_R_total_loss_list.append(train_R_total_loss)
        end = time.perf_counter() 
        elapsed = end-start
    print("R losses: ", train_R_total_loss_list)
    train_R_total_loss_mean = np.mean(np.array([ t.detach().cpu().numpy() for t in train_R_total_loss_list]))
    print("train_R_total_loss_mean: " , train_R_total_loss_mean)
    logger.info(f"Total loss for R: {train_R_total_loss_mean}")
    for i in tqdm(range(num_checkpoint_R)):
        start = time.perf_counter()
        time.sleep(1)
        train_x, train_y = next(train_loader)
        train_R_affine_loss, train_R_output = R_affine_train(train_y, train_x)
        #wtr.check_stop()
        train_R_affine_loss_list.append(train_R_affine_loss)
        end = time.perf_counter() 
        elapsed = end-start
    train_R_affine_loss_mean = np.mean(np.array([ t.detach().cpu().numpy() for t in train_R_affine_loss_list]))
    print("train_R_affine_loss_mean: " , train_R_affine_loss_mean)
    ###############################################################################################################
    # round-end validation
    ###############################################################################################################
    valid_G_total_loss_mean, valid_G_l1_loss_mean, valid_G_ssim_mean, valid_G_psnr_mean, \
        vaild_G_ncc_mean, valid_G_ncc_std, valid_D_real_loss_mean, valid_D_fake_loss_mean, \
        valid_R_total_loss_mean, valid_R_affine_loss_mean = \
        run_validation(vc, iter_D_count, valid_loader, D_test_step, G_test_step, R_test_step)#TODO create a validation
    logger.info(f"------------------------VALIDATION------------------------")
    print("Losses: ")
    print("valid_G_total_loss_mean: " , valid_G_total_loss_mean)
    logger.info(f"Validation G total loss: {valid_G_total_loss_mean}")
    print("valid_G_l1_loss_mean: " , valid_G_l1_loss_mean)
    logger.info(f"Validation G L1 loss: {valid_G_l1_loss_mean}")
    print("valid_G_ssim_mean: " , valid_G_ssim_mean)
    logger.info(f"Validation G SSIM loss: {valid_G_ssim_mean}")
    print("valid_G_psnr_mean: " , valid_G_psnr_mean)
    logger.info(f"Validation G PSNR loss: {valid_G_psnr_mean}")
    print("vaild_G_ncc_mean: " , vaild_G_ncc_mean)
    logger.info(f"Validation G NCC loss mean: {vaild_G_ncc_mean}")
    print("valid_G_ncc_std: " , valid_G_ncc_std)
    logger.info(f"Validation G NCC loss std: {valid_G_ncc_std}")
    print("valid_D_real_loss_mean: " , valid_D_real_loss_mean)
    logger.info(f"Validation D real loss: {valid_D_real_loss_mean}")
    print("valid_D_fake_loss_mean: " , valid_D_fake_loss_mean)
    logger.info(f"Validation D fake loss: {valid_D_fake_loss_mean}")
    print("valid_R_total_loss_mean: " , valid_R_total_loss_mean)
    logger.info(f"Validation R total loss: {valid_R_total_loss_mean}")
    print("valid_R_affine_loss_mean: " , valid_R_affine_loss_mean)
    logger.info(f"Validation R affine loss: {valid_R_affine_loss_mean}")
    
    torch.save(model_G.state_dict(), tc.model_path + f'/model_G_latest.pth')
    torch.save(model_D.state_dict(), tc.model_path + f'/model_D_latest.pth')
    torch.save(model_R.state_dict(), tc.model_path + f'/model_R_latest.pth')
    torch.save(model_R_affine.state_dict(), tc.model_path + f'/model_R_affine_latest.pth')
#cleanup()

#stop_tracing()
#print_memory_usage_by_function()

