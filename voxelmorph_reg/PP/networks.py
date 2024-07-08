import torch
import torch.nn as nn
import torch.functional as F

#The residual network used with MLP

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        if kernel_size == 3:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=strides)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=strides)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, X):
        Y = F.relu((self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
            X = self.bn2(X)
        Y += X
        return F.relu(Y)
    
    
# The overall MLP netowork architecture : One residual layer, and a fully connected layer

class MLP_resnet(nn.Module):
    def __init__(self):
        super(MLP_resnet, self).__init__()
        self.resnet = Residual(300, 128, 1, 0, True) # not sure about the residual part
        self.fc1 = nn.Linear(128, 3)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1).permute(0, 2, 1)
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    
# registration model --- VxmDense : Voxelmorphe non-rigid dvf based registration on already roughly aligned images
"""
model_registration = VxmDense(
    inshape=(256, 256),
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
"""
