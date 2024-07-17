import torch
import torch.nn as nn
import os
import sys
# aligners_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
# sys.path.append(aligners_path)

from aligners.layers import SpatialTransformer
#from GAN_torch.losses_for_gan import NCC

# dataloader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# sys.path.append(dataloader_path)
import math
import torch.nn.functional as F
import numpy as np
from IQA_pytorch import DISTS


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None ,eps = 1e-3):
        self.win = win
        self.eps = eps
    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [20] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(y_true.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = torch.maximum(cross, torch.tensor(self.eps).to(cross.device))
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = torch.maximum(I_var, torch.tensor(self.eps).to(I_var.device))
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = torch.maximum(J_var, torch.tensor(self.eps).to(J_var.device))

        cc = cross * cross / (I_var * J_var + 1e-5)
        
        return 1-torch.mean(cc.view(cc.size(0), -1), dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dim):
        super(ConvBlock, self).__init__()
        if dim == 2:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding = 'same')
            self.pool = nn.MaxPool2d(2)
            self.act = nn.LeakyReLU(0.2)
            
        elif dim == 3:
            self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=3, padding = 'same')
            self.pool = nn.MaxPool2d(2)
            self.act = nn.LeakyReLU(0.2)
    def forward(self,x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.act(x)
        return x    
            


class AffineCNN(nn.Module):     #in_shape=[H,W]
    def __init__(self, in_shape = None, nb_features = [4, 8, 16, 32, 64], nb_levels = None, max_pool = 2, feat_mult = 1, src_feats = 1, trg_feats = 1):
        super(AffineCNN, self).__init__()
        self.dim = len(in_shape)
        
        
        self.list_mod = nn.ModuleList()
        in_channel = src_feats + trg_feats
        if self.dim == 2:
            for i in range(len(nb_features)):
                self.list_mod.append(ConvBlock(in_channel, nb_features[i], self.dim))
                in_channel = nb_features[i]
        elif self.dim == 3:
            for i in range(len(nb_features)):
                self.list_mod.append(ConvBlock(in_channel, nb_features[i], self.dim))
                in_channel = nb_features[i]
        self.flatten = nn.Flatten()
        if self.dim == 2:
            self.dense = nn.Linear(nb_features[-1]*(in_shape[0]//32)*(in_shape[1])//32, 6)
        elif self.dim ==3:
            self.dense = nn.Linear(nb_features[-1]*(in_shape[0]//32)*(in_shape[1]//32)*(in_shape[2]//32), 12)
        self.spatial = SpatialTransformer() 
    def forward(self,src, tgt):     #src is stained greyscale pic; tgt is notstained greyscale pic  shape=[B,1,H,W]
        x = torch.cat([src, tgt], dim = 1)
        for mod in self.list_mod:
            x = mod(x)
        x = self.flatten(x)
        x = self.dense(x)
        #saptial transform
        #print(src.shape)
        src = src.permute(0,2,3,1)
        #print(x.shape)
        moved_image_tensor, disp_tensor = self.spatial([src, x])            #b,h,w,1
        return moved_image_tensor, disp_tensor
    def get_registration_model(self):
        return nn.Sequential(*self.conv_blocks, self.flatten, self.dense)

    def register(self, src, trg):
        with torch.no_grad():
            return self.get_registration_model()(torch.cat([src, trg], dim=1))
                
    def apply_transform(self, src, trg, img):
        with torch.no_grad():
            params = self.register(src, trg)
            return self.spatial(img, params)[0]

def loss_net(moved_out, fixed, model, train = True):
    #print(moved_out.shape, fixed.shape)
    mse_loss = nn.MSELoss()
    l2_loss = mse_loss(moved_out, fixed)
    moving_transformed_clipped = torch.clamp(moved_out, 0,1)
    fixed_clipped = torch.clamp(fixed, 0, 1)
    #ncc = NCC(eps = 1e-3)
    if train:
        R_structure_loss = model(moving_transformed_clipped, fixed_clipped, as_loss = True)
    else:
        R_structure_loss = model(moving_transformed_clipped, fixed_clipped, as_loss = False)      
        
    total_loss = 2.0 * l2_loss + 1.0 * R_structure_loss
    
    return total_loss, l2_loss, R_structure_loss

