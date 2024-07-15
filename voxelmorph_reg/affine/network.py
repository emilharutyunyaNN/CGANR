import torch
import torch.nn as nn
import os
import sys
# aligners_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
# sys.path.append(aligners_path)

from aligners.layers import SpatialTransformer
from GAN_torch.losses_for_gan import NCC

# dataloader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# sys.path.append(dataloader_path)


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
            return self.spatial_transformer(img, params)[0]

def loss_net(moved_out, fixed):
    #print(moved_out.shape, fixed.shape)
    mse_loss = nn.MSELoss()
    l2_loss = mse_loss(moved_out, fixed)
    moving_transformed_clipped = torch.clamp(moved_out, 0,1)
    fixed_clipped = torch.clamp(fixed, 0, 1)
    ncc = NCC(win = 20, eps = 1e-3)
    R_structure_loss = torch.mean(ncc.loss(y_true=fixed_clipped, y_pred = moving_transformed_clipped))      
        
    total_loss = 1.0 * l2_loss + 1.0 * R_structure_loss
    
    return total_loss, l2_loss, R_structure_loss

