from stn_affine import spatial_transform_network, affine_transform
#from utils import affine_to_shift, batch_affine_to_shift
#from layers import SpatialTransformer


"""
currently affine_transform is used as a combination of torch library methods to apply the affine 
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


########################################################
# Helper functions
########################################################
def conv_block(x_in, nf, strides=1):
    """
    Specific convolution module including convolution followed by leakyrelu.
    This block assumes the input is in the format (batch, channels, [depth,] height, width).
    """
    ndims = len(x_in.shape) - 2
    assert ndims in [1, 2, 3], f"ndims should be one of 1, 2, or 3. Found: {ndims}"

    # Select the appropriate Convolutional layer based on the number of dimensions
    if ndims == 1:
        Conv = nn.Conv1d
    elif ndims == 2:
        Conv = nn.Conv2d
    elif ndims == 3:
        Conv = nn.Conv3d

    # Apply convolution
    conv_layer = Conv(in_channels=x_in.shape[1], out_channels=nf, kernel_size=3, padding='same',
                      stride=strides, bias=False)  # Assuming no bias to match 'he_normal'
    x_out = conv_layer(x_in)

    # Apply LeakyReLU activation
    x_out = F.leaky_relu(x_out, negative_slope=0.2)

    return x_out


def conv_block_v2(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(x_in.shape) - 2
    assert ndims in [1,2,3], f"ndims should be one of 1, 2, or 3. Found: {ndims}"
    
    if ndims == 1:
        Conv1 = nn.Conv1d
        Conv2 = nn.Conv1d
    elif ndims == 2:
        Conv1 = nn.Conv2d
        Conv2 = nn.Conv1d
    elif ndims == 3:
        Conv1 = nn.Conv3d
        Conv2 = nn.Conv1d
        
    conv1_layer = Conv1(x_in.shape[1], nf, kernel_size=3, padding='same',
                      stride=strides, bias=False)  
    x_mid = conv1_layer(x_in)
    x_mid = F.leaky_relu(x_mid, negative_slope=0.2)
    
    conv2_layer = Conv2(x_mid.shape[1], nf, kernel_size=3, padding='same',
                      stride=1, bias=False)
    x_out = conv2_layer(x_mid)
    x_out = F.leaky_relu(x_out,negative_slope=0.2)

    return x_out    


def conv_block_v2_residual(x_in, nf):
    """
    Specific convolution module including convolution followed by leakyrelu.
    Two 3x3 stride 1 CONV, followed by one 2x2 average pooling.
    This block assumes the input is in the format (batch, channels, [depth,] height, width).
    """
    ndims = len(x_in.shape) - 2
    assert ndims in [1, 2, 3], f"ndims should be one of 1, 2, or 3. Found: {ndims}"

    # Select the appropriate Convolutional and Pooling layer based on the number of dimensions
    if ndims == 1:
        Conv = nn.Conv1d
        Pool = nn.AvgPool1d
        pool_size = 2
    elif ndims == 2:
        Conv = nn.Conv2d
        Pool = nn.AvgPool2d
        pool_size = (2, 2)
    elif ndims == 3:
        Conv = nn.Conv3d
        Pool = nn.AvgPool3d
        pool_size = (2, 2, 2)

    # First convolution + LeakyReLU
    conv1 = Conv(in_channels=x_in.shape[1], out_channels=nf, kernel_size=3, padding=1)
    x_mid = F.leaky_relu(conv1(x_in), negative_slope=0.2)

    # Second convolution
    conv2 = Conv(in_channels=nf, out_channels=nf, kernel_size=3, padding=1)
    x_out = conv2(x_mid)

    # Residual connection
    if x_in.shape[1] != nf:
        # Adjust x_in if the number of channels is not equal to nf by padding channels
        padding = (0, 0) * ndims + (0, nf - x_in.shape[1])
        x_in_padded = F.pad(x_in, padding, "constant", 0)
    else:
        x_in_padded = x_in

    x_out = x_out + x_in_padded  # element-wise addition
    x_out = F.leaky_relu(x_out, negative_slope=0.2)

    # Average pooling
    pooling = Pool(pool_size)
    x_out = pooling(x_out)

    return x_out
#----------------------------------------------------------------------------
"""
Helping functions for residual affine transformation model
"""

class CustomPad(nn.Module):
    """
        this should be used to add padding to the channel dimension
    """
    def __init__(self, padding, mode='constant', value=0):
        super(CustomPad, self).__init__()
        self.padding = padding
        self.mode = mode
        self.value = value

    def forward(self, x):
        # Padding in PyTorch for 4D tensor: (padding_width_left, padding_width_right, padding_height_top, padding_height_bottom, padding_channel_start, padding_channel_end)
        return F.pad(x, self.padding, mode=self.mode, value=self.value)

class ConvBlockV2Residual(nn.Module):
    def __init__(self, dims, in_channels, nf):
        super(ConvBlockV2Residual, self).__init__()
        self.nf = nf
        
        if dims == 1:
           # print("1")
            self.model = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=nf, kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv1d(in_channels=nf, out_channels=nf, kernel_size=3, padding=1, stride=1),
            )
            self.pool = nn.AvgPool1d(kernel_size=2)
        elif dims == 2:
            #print("2")
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, padding=1, stride=1),
            )
            self.pool = nn.AvgPool2d(kernel_size=2)
        elif dims == 3:
            #print("3")
            self.model = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=nf, kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv3d(in_channels=nf, out_channels=nf, kernel_size=3, padding=1, stride=1),
            )
            self.pool = nn.AvgPool3d(kernel_size=2)
        else:
            raise ValueError("dims should be 1, 2, or 3")

        self.padding_layer = CustomPad(padding=(0, 0) * dims + (0, self.nf - in_channels), mode='constant', value=0)

    def forward(self, x_in):
        #print("x_in type: ", x_in.type())
        x_in = x_in.to(torch.float32)
        #print("x_in type: ", x_in.type())
        x_out = self.model(x_in)

        if x_in.shape[1] != self.nf:
            x_in_padded = self.padding_layer(x_in)
        else:
            x_in_padded = x_in

        x_out = x_out + x_in_padded  # residual connection
        x_out = F.leaky_relu(x_out, negative_slope=0.2)
        x_out = self.pool(x_out)
        return x_out
    
    
#-------------------------------------------------------------------
import torch.nn.init as init 
class ConvLayer(nn.Module):
    """
        Helping class for the aligner_unet_cvpr2018_v3 class
    """
    
    def __init__(self):
        super(ConvLayer, self).__init__()
        # Using LazyConv2d to defer input channel size specification
        self.conv = nn.LazyConv2d(64, kernel_size=3, stride=2, padding=1, bias=False)
        self.init_weights()

    def forward(self, x):
        return self.conv(x)

    def init_weights(self):
        # Initialize the weights with a random normal distribution as per Keras specification
        init.normal_(self.conv.weight, mean=0.0, std=1e-5) 
        

######################################
######## MAIN LAYERS #################
######################################

# Affine transformation without residual block
class Unet_core_v3(nn.Module):
    def __init__(self, vol_size, enc_nf, dec_nf, full_size=True, src_feats=3, tgt_feats=3):
        super(Unet_core_v3, self).__init__()
        self.encoders = nn.ModuleList()
        self.full_size = full_size
        self.src_feats = src_feats
        self.tgt_feats = tgt_feats
        self.dims = len(vol_size) 
        initial_channels = src_feats + tgt_feats
        
        # Adding encoder layers
        for out_channels in enc_nf:
            self.encoders.append(self.conv_block(initial_channels, out_channels, strides=2))
            initial_channels = out_channels

    def conv_block(self, in_channels, out_channels, strides=1):
        """
        This block will compute 'same' padding automatically for stride 1.
        """
        # Padding calculation for 'same' padding
        padding = 1 if strides == 1 else 0

        # Choose the appropriate convolution layer
        if self.dims ==1:
            Conv = nn.Conv1d
        elif self.dims ==2:
            Conv = nn.Conv2d  # Assuming 2D, modify if using 1D or 3D data
        elif self.dims ==3:
            Conv = nn.Conv3d
            
        conv_layer = Conv(in_channels, out_channels, kernel_size=3, padding=padding, stride=strides, bias=False)

        # Creating a sequential block of convolution followed by activation
        block = nn.Sequential(
            conv_layer,
            nn.LeakyReLU(0.2)
        )
        return block

    def forward(self, src, tgt):
        x_in = torch.cat((src, tgt), dim=1)
        x_enc = [x_in]
        
        for encoder in self.encoders:
            x_enc.append(encoder(x_enc[-1]))
        
        return x_enc[-1]
 

class Aligner_unet_cvpr2018_v3(nn.Module):
    """
    UNet with affine translation prediction stemming from the bottleneck representation.
    """
    def __init__(self,vol_size, enc_nf, dec_nf, full_size=True, indexing='ij'):
        super(Aligner_unet_cvpr2018_v3, self).__init__()
        
        self.dims = len(vol_size)
        self.unet = Unet_core_v3(vol_size, vol_size, enc_nf, dec_nf, full_size=full_size)

        self.flow = ConvLayer()
        self.pool = nn.AvgPool2d((8,8))
        self.flatten = nn.Flatten()
        
        self.final = nn.LazyLinear(2)
        
    def forward(self, moving, fixed):
        src = moving
        tgt = fixed
        bottleneck = self.unet(src, tgt)
        flow = self.flow(bottleneck)
        flow = self.pool(flow)
        flow = self.flatten(flow)
        flow = self.final(flow)
        flow_affine = torch.repeat_interleave(flow, repeats=[3, 3], dim=1)  # repeats each element 3 times (Batch, 6)

        # Step 2: Element-wise multiplication to clear out other entries
        mask = torch.tensor([0., 0., 1., 0., 0., 1.], dtype=torch.float32)
        flow_affine = flow_affine * mask.view(1, -1)  # reshape mask for broadcasting

        # Step 3: Add constants to fill in an identity transform
        identity = torch.tensor([1., 0., 0., 0., 1., 0.], dtype=torch.float32)
        flow_affine = flow_affine + identity.view(1, -1) 
        flow_affine = flow_affine.view(flow_affine.shape[0],2 ,3)
        #moving_transformed = spatial_transform_network(src, flow_affine)
        #alternative
        moving_transformed = affine_transform(src, flow_affine)
        return moving_transformed, flow_affine
    
# Affine transformation with a residual block    

class Unet_core_v4(nn.Module):
    def __init__(self, vol_size, enc_nf, dec_nf, full_size=True, src_feats=3, tgt_feats=3):
        super(Unet_core_v4, self).__init__()
        self.encoders = nn.ModuleList()
        self.full_size = full_size
        self.src_feats = src_feats
        self.tgt_feats = tgt_feats
        self.dims = len(vol_size)
        initial_channels = src_feats + tgt_feats

        # Adding encoder layers
        for out_channels in enc_nf:
            self.encoders.append(self.conv_block_v2(initial_channels, out_channels, strides=2))
            initial_channels = out_channels

    def conv_block_v2(self, in_channels, out_channels, strides=1):
        """
        This block will compute 'same' padding automatically for stride 1.
        """
        padding = 1 if strides == 1 else 0

        # Choose the appropriate convolution layer
        Conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[self.dims]

        conv_layer1 = Conv(in_channels, out_channels, kernel_size=3, padding=padding, stride=strides, bias=False)
        conv_layer2 = Conv(out_channels, out_channels, kernel_size=3, padding=padding, stride=1, bias=False)  # No stride change for the second layer

        # Creating a sequential block of convolution followed by activation
        block = nn.Sequential(
            conv_layer1,
            nn.LeakyReLU(0.2),
            conv_layer2,
            nn.LeakyReLU(0.2)
        )
        return block

    def forward(self, src, tgt):
        x_in = torch.cat((src, tgt), dim=1)
        x_enc = [x_in]

        for encoder in self.encoders:
            x_enc.append(encoder(x_enc[-1]))

        return x_enc[-1]
    
    
class Unet_core_v4_residual(nn.Module):
    def __init__(self, vol_size, enc_nf, dec_nf, full_size=True, src_feats=3, tgt_feats=3):
        super(Unet_core_v4_residual, self).__init__()
        self.encoders = nn.ModuleList()
        self.full_size = full_size
        self.src_feats = src_feats
        self.tgt_feats = tgt_feats
        self.dims = len(vol_size)
        initial_channels = src_feats + tgt_feats
        self.pool = {1: nn.AvgPool1d, 2: nn.AvgPool2d, 3: nn.AvgPool3d}[self.dims]
        # Adding encoder layers
        for out_channels in enc_nf:
            self.encoders.append(ConvBlockV2Residual(self.dims, initial_channels, out_channels))
            initial_channels = out_channels

    
    def forward(self, src, tgt):
        x_in = torch.cat((src, tgt), dim=1)
        #print("*** ", x_in.type())
        x_enc = [x_in]
    
        for encoder in self.encoders:
           # print("-------", x_enc[-1].type())
            x_enc.append(encoder(x_enc[-1]))
      
        return x_enc[-1]
    
    
    
class AlignerUNetCVPR2018V4(nn.Module):
    def __init__(self, vol_size, enc_nf, dec_nf, src_feats = 3, tgt_feats = 3, full_size=True, indexing='ij', loss_mask=False, loss_mask_from_prev_cascade=False):
        super(AlignerUNetCVPR2018V4, self).__init__()
        self.vol_size = vol_size
        self.enc_nf = enc_nf
        self.dec_nf = dec_nf
        self.full_size = full_size
        self.indexing = indexing
        self.loss_mask = loss_mask
        self.loss_mask_from_prev_cascade = loss_mask_from_prev_cascade

        # Define the UNet core
        self.unet_model = Unet_core_v4_residual(vol_size, enc_nf, dec_nf, full_size=full_size, src_feats=src_feats, tgt_feats=tgt_feats)
        
        # Define the convolution and pooling layers for the affine transformation prediction
        self.flow_conv = nn.Conv2d(in_channels=enc_nf[-1], out_channels=64, kernel_size=3, padding=1, stride=2)
        pooling_window_size = vol_size[0] // (2 ** (len(enc_nf) + 1))
        self.flow_pool = nn.AvgPool2d((pooling_window_size, pooling_window_size))
        self.flow_fc = nn.Linear(64, 4)

    def forward(self, moving, fixed):
       
        #print("moving and fixed: ", moving, fixed)
        if self.loss_mask:
            print("****")
            if self.loss_mask_from_prev_cascade:
                print("----")
                tgt = moving[:, :-1, :, :]  # Use the first channels as the target
                train_loss_mask = moving[:, -1:, :, :]  # Use the last channel as the loss mask
                bottleneck_repr = self.unet_model(tgt, fixed)
            else:
                print("////")
                # Initialize a new mask
                train_loss_mask = torch.ones((moving.size(0), 1, self.vol_size[0] - 4, self.vol_size[1] - 4), device=moving.device)
                paddings = (2, 2, 2, 2)  # Padding to match the volume size
                train_loss_mask = F.pad(train_loss_mask, paddings, mode="constant", value=1)
                bottleneck_repr = self.unet_model(moving, fixed)
        else:
            #print("++++")
            bottleneck_repr = self.unet_model(moving, fixed)
            #print("Bottleneck: ", bottleneck_repr)
            assert not torch.isnan(bottleneck_repr).any(), "NaN detected in bottleneck during training"
       # print("bottlenck: ", bottleneck_repr.shape)
        flow = self.flow_conv(bottleneck_repr)
        #print("flow1: ", flow)
        #print("flow: ", flow.shape)
        flow = self.flow_pool(flow)
        #print("flow2: ", flow)
        #print("flow after pool: ", flow.shape)
        flow = flow.view(flow.size(0), -1)
       # print("flow3: ", flow)
        #print("flow view: ", flow.shape)
        flow = self.flow_fc(flow)
        #print("flow4: ", flow)
        #print("flow final: ", flow.shape)
        
        # Create affine transformation matrix
        flow_affine = torch.repeat_interleave(flow, repeats=torch.tensor([2, 1, 2, 1], device=flow.device), dim=1)  # (Batch, 6)
        #print("flow affine: ", flow_affine.shape)
        
        flow_affine = flow_affine * torch.tensor([0., 1., 1., 1., 0., 1.], device=flow.device)
        #print("flow_affine mult: ", flow_affine.shape)
        flow_affine = flow_affine + torch.tensor([1., 0., 0., 0., 1., 0.], device=flow.device)
        #print("flow_affine add: ", flow_affine.shape)
        # Apply affine transformation to moving image
        #print("affine: ", flow_affine)
        if self.loss_mask:
            print("&&&&")
            moving_for_warp = torch.cat([moving, train_loss_mask], dim=1)
        else:
            moving_for_warp = moving

        #grid = F.affine_grid(flow_affine.view(-1, 2, 3), moving.size(), align_corners=True)
        #moving_transformed = F.grid_sample(moving_for_warp, grid, mode='bilinear', align_corners=True)
        #moving_for_warp is 1,3,128,128 
        #print("moving: ", moving_for_warp.shape)
        moving_for_warp = moving_for_warp.permute(0,2,3,1)
        #print("moving: ", moving_for_warp.shape)
        #moving_transformed = spatial_transform_network(moving_for_warp, flow_affine)
        #alternative
        flow_affine = flow_affine.view(flow_affine.shape[0],2,3)
        moving_transformed = affine_transform(moving_for_warp, flow_affine)
        assert not torch.isnan(moving_transformed).any(), "NaN detected in moving transformed during training"
        moving_transformed = moving_transformed.permute(0,3,1,2)
        return moving_transformed, flow_affine
    
    
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
#                                 ################ TESTING ##################

##############################################################################################################################################################


def test_custom_pad():
    # Correct padding order: (padding_depth_start, padding_depth_end, padding_height_start, padding_height_end, padding_width_start, padding_width_end)
    # We need to add padding to the channel dimension, which means padding at the end of the tuple.
    padding = (0, 0, 0, 0, 0, 2)  # (depth, height, width, channels)
    custom_pad = CustomPad(padding=padding, mode='constant', value=1)
    
    x = torch.randn(1, 3, 4, 4)  # Example input tensor
    padded_x = custom_pad(x)
    #print(padded_x.shape)
    assert padded_x.shape == (1, 5, 4, 4), "CustomPad test failed!"  # Channels should be 5 (3 + 2)
    #print("CustomPad test passed!")

def test_conv_block_v2_residual():
    x = torch.randn(1, 3, 16, 16)  # Example input tensor
    
    conv_block = ConvBlockV2Residual(dims=2, in_channels=3, nf=64)
    output = conv_block(x)
    print(output.shape)
    assert output.shape == (1, 64, 8, 8), "ConvBlockV2Residual test failed!"
    print("ConvBlockV2Residual test passed!")

def test_aligner_unet_cvpr2018_v4():
    vol_size = (128, 128)
    enc_nf = [64, 128, 256]
    dec_nf = [256, 128, 64]
    
    model = AlignerUNetCVPR2018V4(vol_size, enc_nf, dec_nf, loss_mask=False, loss_mask_from_prev_cascade=False)
    src = torch.randn((1, 3, 128, 128))  # Example source tensor
    tgt = torch.randn((1, 3, 128, 128))  # Example target tensor
    
    output, flow_affine = model(src, tgt)
    print(output.shape)
    assert output.shape == (1, 3, 128, 128), "AlignerUNetCVPR2018V4 output shape test failed!"
    assert flow_affine.shape == (1, 2, 3), "AlignerUNetCVPR2018V4 flow_affine shape test failed!"
    print("AlignerUNetCVPR2018V4 test passed!")

def test_spatial_transform_network():
    B, H, W, C = 1, 5, 5, 3
    img = torch.rand(B, H, W, C)
    theta = torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(B, 1)  # Identity transformation
    
    output = spatial_transform_network(img, theta)
    print("input: ", img)
    print("output: ", output)
    assert torch.allclose(img, output, atol=1e-6), "SpatialTransformerNetwork test failed!"
    print("SpatialTransformerNetwork test passed!")

"""if __name__ == "__main__":
    test_custom_pad()
    test_conv_block_v2_residual()
    test_aligner_unet_cvpr2018_v4()
    test_spatial_transform_network()"""