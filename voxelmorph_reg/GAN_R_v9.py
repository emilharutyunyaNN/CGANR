from GAN_torch.att_unet_2d import att_unet_2d
from GAN_torch.discriminator_2d import discriminator_2d
from GAN_torch.losses_for_gan import *
import torch
import glob
torch.cuda.empty_cache()
import sys
"""
works with batch_utils 8 but I don't think is going to have the same effect on training because 
it doesn't load images from another big image until it is done with the current one
there's diversity but not as much as in the original
"""


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
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
import cv2
import gc
import sys
from aligners.aligner import Aligner_unet_cvpr2018_vJX
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
from dataloader.batch_utils_v8 import *
import matplotlib.pyplot as plt
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
import tracemalloc
import psutil

##### MEMORY OPERATIONS #######
def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / (1024 ** 2):.2f} MB")

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()
    print("Cleared memory")

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
import random
import torchvision.transforms as transforms

"""
V7 - Addin gaussian blurr on the lables (V6 doesn't go to mode collapse as much as it was trained it reaches >15 psnr and 0.15 ssim and ncc )
"""


#world_size = torch.cuda.device_count()
#mp.spawn(main, args = (world_size,dataset,), nprocs=world_size, join=True)

# Directly test dataset iteration without DataLoader
#for i in range(len(dataset)):
#   band_patches, rgb_patches = dataset[i]
#  print(f"Item {i}: band_patches shape: {band_patches.shape}, rgb_patches shape: {rgb_patches.shape}")



#train_loader = iter(train_loader)
#print(next(dataloader))
#valid_loader = iter(validation_loader)



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

    #tc.image_path = "/home/hkhz/daihui/Training/target/*.mat" # path for training data 
    #vc.image_path = "/home/hkhz/daihui/Validation/target/*.mat" # path for validation data

    tc.image_path = "/home/hkhz/daihui/Training/target/*.jpg"
    vc.image_path = "/home/hkhz/daihui/Validation/target/*.jpg"
    
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


def run_validation(vc, cur_iter, iterator_valid_bl, D_test_step,G_test_step, R_test_step, use_tqdm = True):
    valid_G_total_loss_list, valid_G_l1_loss_list, valid_G_ssim_list, valid_G_psnr_list, valid_G_ncc_list, \
    valid_D_real_loss_list, valid_D_fake_loss_list, valid_R_total_loss_list = [], [], [], [], [], [], [], []
    valid_idx_iterator = tqdm(range(vc.q_limit)) if use_tqdm else range(vc.q_limit)
    #print(valid_idx_iterator)
    
    for i in valid_idx_iterator:
        if not use_tqdm:
            print("Running Validation: {}/{}".format(i+1, vc.q_limit), end='\r')
            if i == vc.q_limit - 1:
                print()
        #valid_y, valid_x = next(iterator_valid_bl)    
        valid_x, valid_y = next(iterator_valid_bl)
        # = data_v['input']
        # = data_v['target']  
        valid_G_total_loss, valid_G_dis_loss, valid_G_l1_loss, valid_G_ssim, valid_G_psnr, valid_G_ncc = G_test_step(
            valid_x, valid_y, 3)
        valid_D_total_loss, valid_D_real_loss, valid_D_fake_loss, valid_G_output = D_test_step(valid_x, valid_y)
        valid_R_total_loss, _, valid_R_output = R_test_step(valid_x, valid_y)
        #print("R test: ", valid_R_total_loss)
        #print("done")
        #if i < 5:
          # for j in range(vc.batch_size):
              #  if tc.label_channels == 1:
              #      plt.imsave(tc.model_path + f'/output/iter={cur_iter}_sample={i * vc.batch_size + j}_outputG.jpg',
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
    #print(valid_R_total_loss_list)
    valid_G_total_loss_mean = np.mean(np.array([t.cpu().numpy() for t in valid_G_total_loss_list]))
    valid_G_l1_loss_mean = np.mean(np.array([t.cpu().numpy() for t in valid_G_l1_loss_list]))
    valid_G_ssim_mean = np.mean(np.array([t.cpu().numpy() for t in valid_G_ssim_list]))
    #print("PSNR: ", valid_G_psnr_list)
    valid_G_psnr_mean = np.mean(np.array([t.cpu().numpy() for t in valid_G_psnr_list]))
    valid_G_ncc_mean = np.mean(np.array(valid_G_ncc_list))
    valid_G_ncc_std = np.std(np.array(valid_G_ncc_list))
    valid_D_real_loss_mean = np.mean(np.array([t.cpu().numpy() for t in valid_D_real_loss_list]))
    valid_D_fake_loss_mean = np.mean(np.array([t.cpu().numpy() for t in valid_D_fake_loss_list]))
    valid_R_total_loss_mean = np.nanmean(np.array([t.cpu().numpy() for t in valid_R_total_loss_list]))
    return valid_G_total_loss_mean, valid_G_l1_loss_mean, valid_G_ssim_mean, valid_G_psnr_mean, valid_G_ncc_mean, valid_G_ncc_std, valid_D_real_loss_mean, valid_D_fake_loss_mean, valid_R_total_loss_mean

def default_unet_features():
    nb_features = [
        [8, 16, 16, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]#[128, 128, 128, 64, 32, 16, 16]  # decoder
    ]
    return nb_features
if __name__ == "__main__":
    tc,vc = init_parameters()
    train_images = glob.glob(tc.image_path)
    valid_images = glob.glob(vc.image_path)
    random.shuffle(train_images)
    random.shuffle(valid_images)
    
    
    transform = Augmentation(num_slices=tc.channel_end_index, label_channels=tc.label_channels)

# Create training and validation dataset instances
    train_dataset = InterleavedHDF5Dataset('/home/hkhz/daihui.hdf5', 'Training', tc, transform=transform)
    val_dataset = InterleavedHDF5Dataset('/home/hkhz/daihui.hdf5', 'Validation', vc, transform=None)

    train_dataloader = DataLoader(train_dataset, batch_size=tc.batch_size, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=vc.batch_size, num_workers=2)
   # infinite_train_dataloader = InfiniteDataLoader(train_dataloader)
    #infinite_val_dataloader = InfiniteDataLoader(val_dataloader)
    
    #train_bl = TransformImageBatchLoader(train_images, tc, tc.num_slices, is_testing=False,
                                             # n_parallel_calls=tc.n_threads, q_limit=tc.q_limit,
                                            #  n_epoch=tc.n_shuffle_epoch)
    #valid_bl = TransformImageBatchLoader(valid_images, vc, vc.num_slices, is_testing=True,
                                            #  n_parallel_calls=vc.n_threads, q_limit=vc.q_limit,
                                            #  n_epoch=vc.n_shuffle_epoch)

    
    
    filters = [32,64,128,256, 512]
    #ddp_setup(rank, world_size)
    model_G = att_unet_2d(input_size=(3, 256, 256), filter_num=filters, n_labels=3, rank=1, stack_num_down=3, stack_num_up=3, activation='LeakyReLU',
                            atten_activation='ReLU', attention='add',
                            output_activation=None, batch_norm=True, pool='ave', unpool='bilinear').to(device)
    #model_G = DDP(model_G, device_ids=[rank])

    model_D = discriminator_2d(input_size=(3, 256, 256), filter_num=filters, rank = 1, stack_num_down=1, activation='LeakyReLU',
                                batch_norm=False, pool=False, backbone=None).to(device)
    #model_D = DDP(model_D, device_ids=[rank])

    """model_R = VxmDense(
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
    ).to(device)"""

    model_R = Aligner_unet_cvpr2018_vJX(vol_size=[256,256], enc_nf= tc.nf_enc, dec_nf=tc.nf_dec, gauss_kernal_size=tc.gauss_kernel_size,
                                            flow_clipping=tc.dvf_clipping,
                                            flow_clipping_nsigma=tc.dvf_clipping_nsigma,
                                            flow_thresholding=tc.dvf_thresholding,
                                            flow_thresh_dis=tc.dvf_thresholding_distance,
                                            loss_mask=tc.loss_mask, loss_mask_from_prev_cascade=False).to(device)

    """model_R = VxmDense(
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
    model_R.to(device)"""

    G_optimizer = torch.optim.Adam(model_G.parameters(),lr=1e-4)
    D_optimizer = torch.optim.Adam(model_D.parameters(),lr=1e-5)
    R_optimizer = torch.optim.Adam(model_R.parameters(),lr=1e-4)

    def G_train_step(input_image, target, epoch):
        input_image = input_image.to(device)
        target = target.to(device)
        
        model_G.train()
        model_D.train()
        model_R.eval()
        
        G_optimizer.zero_grad()
        
        G_outputs = model_G(input_image)
        
        # Save generated and target images for debugging
        """ os.makedirs("./voxelmorph_reg/G", exist_ok=True)
        for i, G in enumerate(G_outputs):
            plt.imsave(f"./voxelmorph_reg/G/G_{i+1}.jpg", np.clip(G.permute(1, 2, 0).detach().cpu().numpy(), 0, 1))
        for j, tgt in enumerate(target):
            plt.imsave(f"./voxelmorph_reg/G/tgt_{j+1}.jpg", np.clip(tgt.permute(1, 2, 0).detach().cpu().numpy(), 0, 1))
        """
        assert not torch.isnan(G_outputs).any(), "Tensor contains NaN."

        #with torch.no_grad():
        D_fake_output = model_D(G_outputs)
        
        if epoch > 0:
            #start = time.perf_counter()
            #time.sleep(1)
            
            target_transformed, _, _ = model_R(target, G_outputs)
            assert not torch.isnan(target_transformed).any(), "Tensor contains NaN."
            #end = time.perf_counter()
            #elapsed = end - start
        else:
            target_transformed = target
        
        G_total_loss, G_dis_loss, G_l1_loss = loss_G_perceptual(D_fake_output, G_outputs, target_transformed, tc, epoch)
        #print(f"Epoch: {epoch}, G_total_loss: {G_total_loss.item()}, G_dis_loss: {G_dis_loss.item()}, G_l1_loss: {G_l1_loss.item()}")
        for name, param in model_G.named_parameters():
            if not param.requires_grad:
                print(f"Parameter {name} does not require gradients.")
                
        for param in model_G.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param)
        G_total_loss.backward()
       
        
        G_optimizer.step()
        
        return G_total_loss, G_dis_loss, G_l1_loss, G_outputs, target_transformed

    def G_test_step(input_image, target, epoch):
        input_image, target = input_image.to(device), target.to(device)

        model_G.eval()
        model_D.eval()
        with torch.no_grad():
            G_outputs = model_G(input_image)
            
        #print("G_train: ", G_outputs.shape, input_image.shape, target.shape)
        G_output_clipped_splitted = split_tensor(torch.clamp(G_outputs, 0, 1), tc.case_filtering_x_subdivision,
                                                    tc.case_filtering_y_subdivision)
        target_clipped_splitted = split_tensor(torch.clamp(target,0,1), tc.case_filtering_x_subdivision,
                                                    tc.case_filtering_y_subdivision)
        
        #print("G_train: ", G_output_clipped_splitted.shape, target_clipped_splitted.shape)
        with torch.no_grad():
            D_fake_output = model_D(G_outputs)
        G_total_loss, G_dis_loss, G_l1_loss = loss_G_perceptual(D_fake_output, G_outputs, target, tc, epoch)
        G_ssim = compute_ssim(G_outputs, target) #TODO: check if correct
        G_psnr = compute_psnr(G_outputs,target)
        #print("PSNR: ", G_psnr)
    # print("+++", target_clipped_splitted.shape, G_output_clipped_splitted.shape)
        G_ncc = torch.mean(NCC(win = 20, eps = 1e-3).ncc(target_clipped_splitted, G_output_clipped_splitted))# TODO: check if correct
        return G_total_loss, G_dis_loss, G_l1_loss, G_ssim, G_psnr, G_ncc



    def D_train_step(input_image, target,epoch):
    # train_sampler.set_epoch(epoch)
        model_D.train()
        model_G.train()
        input_image, target = input_image.to(device), target.to(device)
        D_optimizer.zero_grad()
        with torch.no_grad():
            G_outputs = model_G(input_image)
        
        D_real_output = model_D(target)
        D_fake_output = model_D(G_outputs)
        D_total_loss, D_real_loss, D_fake_loss = loss_D(D_real_output, D_fake_output)
        
        D_total_loss.backward()
        
        D_optimizer.step()
    
    
        return D_total_loss.item(), D_real_loss.item(), D_fake_loss.item(), G_outputs

    #CORRECT
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
        model_G.train()
        R_optimizer.zero_grad()
        with torch.no_grad():
            G_outputs = model_G(input_image)
        
        R_outputs = model_R(target, G_outputs)
        #print("-----", R_outputs[0].shape)
        #img = R_outputs[0][0,:,:,:].permute(1,2,0).detach().clone().cpu().numpy() 
        #plt.imsave(f"./voxelmorph_reg/R/epoch_{epoch}_round_{i}.jpg",np.clip(img, 0, 1))
        R_total_loss, R_berhu_loss = loss_R_no_gt(R_outputs, G_outputs, tc)
        #print("R losses: ", R_total_loss, R_berhu_loss)
        R_total_loss.backward()
        R_optimizer.step()
    
        return R_total_loss, R_berhu_loss, R_outputs[0]

    def R_test_step(input_image, target):
        model_G.eval()
        model_R.eval()
        input_image, target = input_image.to(device), target.to(device)
        with torch.no_grad():
            G_outputs = model_G(input_image)
            R_outputs = model_R(target, G_outputs)
        R_total_loss, R_berhu_loss = loss_R_no_gt(R_outputs, G_outputs, tc)
        return R_total_loss, R_berhu_loss, R_outputs

    min_loss = 1e8
    max_psnr = 0
    epoch_begin = tc.epoch_begin
    iter_D_count = 0
    warmstart_first_epoch_elapsed_iters = None
    if tc.iter_begin is not None:
        iter_D_count+=tc.iter_begin
        assert epoch_begin is not None
        warmstart_first_epoch_elapsed_iters = tc.iter_begin
        for i in range(epoch_begin):
            warmstart_first_epoch_elapsed_iters -= max(int(tc.initial_alternate_steps*0.9**i), 500)
    elif epoch_begin is not None:
        for i in range(epoch_begin):
            iter_D_count+=max(int(tc.initial_alternate_steps*0.9**i), 500)

    print("Training from iteration: ", iter_D_count)

    
    for epoch in range(epoch_begin, tc.N_epoch):
        train_loader = iter(train_dataloader)
        valid_loader = iter(val_dataloader)

        print(f'Current iter_D_count: {iter_D_count}')

        train_G_total_loss_list, train_G_l1_loss_list, train_R_total_loss_list = [], [], []

        if warmstart_first_epoch_elapsed_iters is None or warmstart_first_epoch_elapsed_iters == 0:

            num_checkpoint_D = max(int(tc.initial_alternate_steps * 0.9 ** epoch), 500)
            num_checkpoint_R = max(int(tc.initial_alternate_steps * 0.9 ** epoch), 500)
        else:
            num_checkpoint_D = warmstart_first_epoch_elapsed_iters
            num_checkpoint_R = warmstart_first_epoch_elapsed_iters
            warmstart_first_epoch_elapsed_iters = None
        if epoch > 0 or (tc.G_warmstart_checkpoint is None and tc.prev_checkpoint_path is None):        
            print('training G & D ...')
            tc.epoch_filtering_ratio = []

            if tc.case_filtering and epoch >= tc.case_filtering_starting_epoch:
                format_str = "Train G Case Filtering {} for epoch {} with threshold {} and {}x{} subdivision"
                print(format_str.format(tc.case_filtering, epoch, tc.case_filtering_cur_mean- tc.case_filtering_nsigma*tc.case_filtering_cur_stdev, tc.case_filtering_x_subdivision, tc.case_filtering_y_subdivision))
            #num_checkpoint_D=100
            else:
                print("Train G Case Filtering False for epoch {}". format(epoch))
            #num_checkpoint_D = 50
            for i in tqdm(range(num_checkpoint_D)):
                #start = time.perf_counter()
                #time.sleep(1)
                
                NumGen = max(3, int(12 - iter_D_count //4000))
                
                for j in range(NumGen):
                    #targets, input_images = next(train_loader)
                    input_images, targets= next(train_loader)
                     #= data_t['input']
                    #targets = data_t['target']
                    input_images = input_images.to(device)
                    targets = targets.to(device)
                   # print(input_images.shape)
                    #print(targets.shape)
                    #print("----", (targets[0,:,:,:].permute(1,2,0).detach().cpu().numpy()<0).any())
                    #print("----", (input_images[0,:,:,:].permute(1,2,0).detach().cpu().numpy()<0).any())
                    #train_G_total_loss, train_G_dis_loss, train_G_l1_loss, train_G_output, train_R_output = G_train_step(
                    #input_images, targets, epoch)
                    train_G_total_loss, train_G_dis_loss, train_G_l1_loss, train_G_output, train_R_output = G_train_step(
                    input_images, targets, epoch)
                    train_G_total_loss_list.append(train_G_total_loss)
                    train_G_l1_loss_list.append(train_G_l1_loss)
                
                if i%25 ==0:
                    os.makedirs(f"./voxelmorph_reg/output_image/iter_{i}", exist_ok = True)
                    #print("**: ", targets[0,:,:,:].shape)
                    #print(train_G_output[0,:,:,:].shape)
                    #print("----", (targets[0,:,:,:].permute(1,2,0).detach().cpu().numpy()<0).any())
                    plt.imsave(f"./voxelmorph_reg/output_image/iter_{i}/train_epoch={epoch}_sample_output_G.jpg", np.clip(train_G_output[0,:,:,:].permute(1,2,0).detach().clone().cpu().numpy(), 0, 1))
                    plt.imsave(f"./voxelmorph_reg/output_image/iter_{i}/train_epoch={epoch}_sample_output_R.jpg", np.clip(train_R_output[0,:,:,:].permute(1,2,0).detach().clone().cpu().numpy(), 0, 1))
                    plt.imsave(f"./voxelmorph_reg/output_image/iter_{i}/train_epoch={epoch}_sample_target.jpg", targets[0,:,:,:].permute(1,2,0).detach().clone().cpu().numpy())
                    plt.imsave(f"./voxelmorph_reg/output_image/iter_{i}/train_epoch={epoch}_sampleinput0.jpg", -input_images[0,0,:,:].detach().clone().cpu().numpy(), cmap = 'gray')
                #print("training fore: ", torch.cuda.memory_allocated())
                #train_y, train_x = next(train_loader)
                train_x, train_y = next(train_loader)
                 #= data_d['input']
                # = data_d['target']
                #print("training after : ", torch.cuda.memory_allocated())
            # print("'fore D: ", train_x.shape, train_y.shape)
                train_D_total_loss, train_D_real_loss, train_D_fake_loss, _ = D_train_step(train_x, train_y, epoch)
                iter_D_count += 1
                #del train_D_total_loss, train_D_real_loss, train_D_fake_loss
            #  print("memory allocated : ", torch.cuda.memory_allocated())
                #end = time.perf_counter() 
                #elapsed = end-start
            #  print("D check:", num_checkpoint_D)
                torch.cuda.empty_cache()
                #######################################################################################################
                # mid-round validation
                #######################################################################################################
                if iter_D_count % tc.valid_steps == 0 and i!= num_checkpoint_D-1:
                    valid_G_total_loss_mean, valid_G_l1_loss_mean, valid_G_ssim_mean, valid_G_psnr_mean, \
                        valid_G_ncc_mean, valid_G_ncc_std, valid_D_real_loss_mean, valid_D_fake_loss_mean, \
                        valid_R_total_loss_mean = run_validation(vc, iter_D_count, valid_loader, D_test_step,
                                                                G_test_step, R_test_step, use_tqdm=False) 
                    
                    train_G_total_loss_mean = np.mean(np.array([ t.detach().cpu().numpy() for t in train_G_total_loss_list]))
                    train_G_l1_loss_mean = np.mean(np.array([ t.detach().cpu().numpy() for t in train_G_l1_loss_list]))
                    train_R_total_loss_mean = np.mean(np.array([ t.detach().cpu().numpy() for t in train_R_total_loss_list]))

                    train_G_total_loss_list, train_G_l1_loss_list, train_R_total_loss_list = [], [], []

                    if tc.case_filtering and tc.case_filtering_recalc_every_eval:
                        tc.case_filtering_curr_mean, tc.case_filtering_curr_stdev = valid_G_ncc_mean, valid_G_ncc_std
                        print("Update filtering mean to {} and stdev to {}".format(valid_G_ncc_mean, valid_G_ncc_std))


                    logger.info(f"------------------------MID-ROUND VALIDATION------------------------")
                    print("Round: ", epoch)
                    print("iter_D_count: ", iter_D_count)
                    print("Losses: ")
                    print("train_G_total_loss_mean: " , train_G_total_loss_mean)
                    logger.info(f"Train G total loss: {train_G_total_loss_mean}")
                    print("train_G_l1_loss_mean: " , train_G_l1_loss_mean)
                    logger.info(f"Train G L1 loss: {train_G_l1_loss_mean}")
                    print("valid_G_ssim_mean: " , valid_G_ssim_mean)
                    print("valid_G_total_loss_mean: " , valid_G_total_loss_mean)
                    logger.info(f"Validation G total loss: {valid_G_total_loss_mean}")
                    print("valid_G_l1_loss_mean: " , valid_G_l1_loss_mean)
                    logger.info(f"Validation G L1 loss: {valid_G_l1_loss_mean}")
                    print("valid_G_ssim_mean: " , valid_G_ssim_mean)
                    logger.info(f"Validation G SSIM loss: {valid_G_ssim_mean}")
                    print("valid_G_psnr_mean: " , valid_G_psnr_mean)
                    logger.info(f"Validation G PSNR loss: {valid_G_psnr_mean}")
                    print("valid_G_ncc_mean: " , valid_G_ncc_mean)
                    logger.info(f"Validation G NCC loss mean: {valid_G_ncc_mean}")
                    print("valid_G_ncc_std: " , valid_G_ncc_std)
                    logger.info(f"Validation G NCC loss std: {valid_G_ncc_std}")
                    print("valid_D_real_loss_mean: " , valid_D_real_loss_mean)
                    logger.info(f"Validation D real loss: {valid_D_real_loss_mean}")
                    print("valid_D_fake_loss_mean: " , valid_D_fake_loss_mean)
                    logger.info(f"Validation D fake loss: {valid_D_fake_loss_mean}")
                    print("valid_R_total_loss_mean: " , valid_R_total_loss_mean)
                    logger.info(f"Validation R total loss: {valid_R_total_loss_mean}")
                    logger.info(f"------------------------MID-ROUND VALIDATION------------------------")
                    
                
                    #train_G_total_loss_list, train_G_l1_loss_list, train_R_total_loss_list = [], [], []
                    #if min_loss - valid_G_l1_loss_mean > tc.min_del or valid_G_psnr_mean > max_psnr \
                    #        or tc.save_every_epoch:
                    #    # tol = 0  # refresh early stopping patience
                    #torch.save(model_G.state_dict,tc.model_path + f'/model_G_iter={iter_D_count}.h5')
                    #torch.save(model_D.state_dict,tc.model_path + f'/model_D_iter={iter_D_count}.h5')
                    #..save(model_R.state_dict, tc.model_path + f'/model_R_iter={iter_D_count}.h5')
                #print("validation done")
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
        #train_G_total_loss_mean = np.mean(np.array([ t.detach().cpu().numpy() for t in train_G_total_loss_list]))
        #train_G_l1_loss_mean = np.mean(np.array([ t.detach().cpu().numpy() for t in train_G_l1_loss_list]))
    # print("Training losses")
    # logger.info("-------------------TRAINING LOSSES-------------------")"""
        print("first done")
        """print("train_G_total_loss_mean: " , train_G_total_loss_mean)
        logger.info(f"Total loss for G: {train_G_total_loss_mean}")
        print("train_G_l1_loss_mean: " , train_G_l1_loss_mean)
        logger.info(f"L1 loss for G: { train_G_l1_loss_mean}")"""
        print('training R ...')
        #num_checkpoint_R = 50
        for i in tqdm(range(num_checkpoint_R)):
            #start = time.perf_counter()
            #time.sleep(1)
            #train_y, train_x = next(train_loader)
            train_x, train_y = next(train_loader)
            # = data_R['input']
            # = data_R['target']
            train_R_total_loss, _, train_R_output = R_train_step(train_x, train_y)
            # TRAIN R Check
            #os.makedirs("./voxelmorph_reg/R", exist_ok= True)
            #plt.imsave(f"./voxelmorph_reg/R/epoch_{epoch}_round_{i}.jpg",np.clip(train_R_output[0,:,:,:].permute(1,2,0).detach().cpu().numpy(), 0, 1))
            #wtr.check_stop()
            train_R_total_loss_list.append(train_R_total_loss)
           #end = time.perf_counter() 
            #elapsed = end-start
            print_memory_usage()
            torch.cuda.empty_cache()

        """print("R losses: ", train_R_total_loss_list)
        train_R_total_loss_mean = np.mean(np.array([ t.detach().cpu().numpy() for t in train_R_total_loss_list]))
        print("train_R_total_loss_mean: " , train_R_total_loss_mean)
        logger.info(f"Total loss for R: {train_R_total_loss_mean}")"""
        ###############################################################################################################
        # round-end validation
        ###############################################################################################################
        valid_G_total_loss_mean, valid_G_l1_loss_mean, valid_G_ssim_mean, valid_G_psnr_mean, \
            valid_G_ncc_mean, valid_G_ncc_std, valid_D_real_loss_mean, valid_D_fake_loss_mean, \
            valid_R_total_loss_mean = \
            run_validation(vc, iter_D_count, valid_loader, D_test_step, G_test_step, R_test_step)
        
        train_R_total_loss_mean = np.mean(np.array([ t.detach().cpu().numpy() for t in train_R_total_loss_list]))
        train_G_total_loss_mean = np.mean(np.array([ t.detach().cpu().numpy() for t in train_G_total_loss_list]))
        train_G_l1_loss_mean = np.mean(np.array([ t.detach().cpu().numpy() for t in train_G_l1_loss_list]))
        train_G_total_loss_list, train_G_l1_loss_list, train_R_total_loss_list = [], [], []

        if tc.case_filtering and tc.case_filtering_recalc_every_eval:
                        tc.case_filtering_curr_mean, tc.case_filtering_curr_stdev = valid_G_ncc_mean, valid_G_ncc_std
                        print("Update filtering mean to {} and stdev to {}".format(valid_G_ncc_mean, valid_G_ncc_std))

        logger.info(f"------------------------VALIDATION------------------------")
        print("Round: ", epoch)
        logger.info(f"Round: {epoch}")
        logger.info(f"Train R total loss: {train_R_total_loss_mean}")
        print("iter_D_count: ", iter_D_count)
        logger.info(f"iter_D_count: {iter_D_count}")
        print("Losses: ")
        print("train_R_total_loss_mean: " , train_R_total_loss_mean)
        logger.info(f"Train R total loss: {train_R_total_loss_mean}")
        print("train_G_total_loss_mean: " , train_G_total_loss_mean)
        logger.info(f"Train G total loss: {train_G_total_loss_mean}")
        print("train_G_l1_loss_mean: " , train_G_l1_loss_mean)
        logger.info(f"Train G L1 loss: {train_G_l1_loss_mean}")
        print("valid_G_total_loss_mean: " , valid_G_total_loss_mean)
        logger.info(f"Validation G total loss: {valid_G_total_loss_mean}")
        print("valid_G_l1_loss_mean: " , valid_G_l1_loss_mean)
        logger.info(f"Validation G L1 loss: {valid_G_l1_loss_mean}")
        print("valid_G_ssim_mean: " , valid_G_ssim_mean)
        logger.info(f"Validation G SSIM loss: {valid_G_ssim_mean}")
        print("valid_G_psnr_mean: " , valid_G_psnr_mean)
        logger.info(f"Validation G PSNR loss: {valid_G_psnr_mean}")
        print("valid_G_ncc_mean: " , valid_G_ncc_mean)
        logger.info(f"Validation G NCC loss mean: {valid_G_ncc_mean}")
        print("valid_G_ncc_std: " , valid_G_ncc_std)
        logger.info(f"Validation G NCC loss std: {valid_G_ncc_std}")
        print("valid_D_real_loss_mean: " , valid_D_real_loss_mean)
        logger.info(f"Validation D real loss: {valid_D_real_loss_mean}")
        print("valid_D_fake_loss_mean: " , valid_D_fake_loss_mean)
        logger.info(f"Validation D fake loss: {valid_D_fake_loss_mean}")
        print("valid_R_total_loss_mean: " , valid_R_total_loss_mean)
        logger.info(f"Validation R total loss: {valid_R_total_loss_mean}")
        
        
        del train_R_total_loss_mean, train_G_total_loss_mean, valid_G_total_loss_mean, valid_G_l1_loss_mean, valid_G_ssim_mean, valid_G_psnr_mean, \
            valid_G_ncc_mean, valid_G_ncc_std, valid_D_real_loss_mean, valid_D_fake_loss_mean, valid_R_total_loss_mean
        del train_G_total_loss_list, train_G_l1_loss_list, train_R_total_loss_list

        print_memory_usage()  # Print memory usage after each epoch
        clear_memory()
        
        
        torch.save(model_G.state_dict,tc.model_path + f'/model_G_latest.h5')
        torch.save(model_D.state_dict,tc.model_path + f'/model_D_latest.h5')
        torch.save(model_R.state_dict,tc.model_path + f'/model_R_latest.h5')
        torch.save(G_optimizer.state_dict(), os.path.join(tc.model_path, 'optimizer_G_latest.pth'))
        torch.save(D_optimizer.state_dict(), os.path.join(tc.model_path, 'optimizer_D_latest.pth'))
        torch.save(R_optimizer.state_dict(), os.path.join(tc.model_path, 'optimizer_R_latest.pth'))
        
        
        """train_loader.reset()  # Reset the loader at the end of each epoch
        train_dataset.reset_generators()
        
        valid_loader.reset()  # Reset the loader at the end of each epoch
        val_dataset.reset_generators()"""