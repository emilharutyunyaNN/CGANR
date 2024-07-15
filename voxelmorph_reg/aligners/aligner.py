import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# third party
import numpy as np
from aligners.layers import SpatialTransformer

def matlab_style_gauss2D(shape=(40, 40), sigma=20):

    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def gausv2(inp, kernel_size=80):
    # kernel_size = 80  # set the filter size of Gaussian filter
    kernel_weights = matlab_style_gauss2D(shape=(kernel_size, kernel_size), sigma=kernel_size//2)  # compute the weights of the filter with the given size (and additional params)

    # assuming that the shape of `kernel_weighs` is `(kernel_size, kernel_size)`
    # we need to modify it to make it compatible with the number of input channels
    in_channels = 2  # the number of input channels
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    kernel_weights = np.repeat(kernel_weights, in_channels, axis=-1)  # apply the same filter on all the input channels
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)  # for shape compatibility reasons

    # define your model...

    # somewhere in your model you want to apply the Gaussian blur,
    # so define a DepthwiseConv2D layer and set its weights to kernel weights
    #g_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
   # g_layer_out = g_layer(inp)  # apply it on the input Tensor of this layer

    # do this BEFORE calling `compile` method of the model
  #  g_layer.set_weights([kernel_weights])
   # g_layer.trainable = False
    #return g_layer_out
#
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
        super(ConvLayer, self).__init__()
        # Set padding to 'same' if required
        #TODO : doesn't work for even kernel sizes
        if padding == 'same':
            self.padding = (kernel_size // 2, kernel_size // 2)  # This only works correctly for odd kernel sizes
        else:
            self.padding = 0

        # Define the convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding
        )
        
        # Initialize weights using He Normal initialization
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')

        # Optional: Initialize bias to zero if there is a bias
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = nn.LeakyReLU(0.2)(x)
        return x
    
import torch.nn.functional as F

class GaussianBlurLayer(nn.Module):
    def __init__(self, channel, kernel_size=80):
        super(GaussianBlurLayer, self).__init__()

        sigma = kernel_size // 2
        kernel_weights = matlab_style_gauss2D(shape=(kernel_size, kernel_size), sigma=sigma)
        if kernel_size % 2 == 1:
            pad = (kernel_size // 2, kernel_size // 2)
        else:
            pad = (kernel_size // 2, kernel_size // 2 - 1)
        
        # Convert kernel_weights to a PyTorch tensor and adjust dimensions
        kernel_weights = torch.from_numpy(kernel_weights).float()  # Convert to tensor and ensure float type
        kernel_weights = kernel_weights.unsqueeze(0).unsqueeze(0)  # Add two dimensions to fit [out_channels, in_channels/groups, height, width]
        kernel_weights = kernel_weights.repeat(channel, 1, 1, 1)  # Repeat across the channel dimension for depthwise convolution

        self.groups = channel
        self.conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=kernel_size,
                              padding='same', dilation=1, groups=self.groups, bias=False)
        
        # Assign the Gaussian kernel weights and set them to be non-trainable
        self.conv.weight.data = kernel_weights
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)
import cv2
import matplotlib.pyplot as plt

"""def main():
    # Parameters for the test
    in_channels = 3  # Example for RGB images
    kernel_size = 20  # Size of Gaussian kernel
    batch_size = 1    # Single image in batch
    image_size = 256  # Spatial size of the test image

    # Create a GaussianBlurLayer instance
    gauss_blur_layer = GaussianBlurLayer(channel=in_channels, kernel_size=kernel_size)

    # Generate a random image tensor [batch_size, channels, height, width]
    #input_tensor = torch.rand(batch_size, in_channels, image_size, image_size)
    img = cv2.imread("/home/hkhz/remote_mnt/filtered_stained_split/stained_patch_0_7_18432_47104_61.jpg", cv2.IMREAD_COLOR)
    # Apply the Gaussian blur layer
    transform = transforms.ToTensor()
    input_tensor = transform(img)  # Correctly apply the transformation
    input_tensor = input_tensor.unsqueeze(0)
    output_tensor = gauss_blur_layer(input_tensor)

    # Display the input and output images to visually confirm the blur effect
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(input_tensor[0].permute(1, 2, 0).numpy(), interpolation='nearest')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output_tensor[0].permute(1, 2, 0).numpy(), interpolation='nearest')
    plt.title('Blurred Image')
    plt.axis('off')

    plt.savefig("transformed.jpg")

if __name__ == "__main__":
    main()"""
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')

        # Optional: Initialize bias to zero if there is a bias
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)


    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class UNet(nn.Module):
    def __init__(self, vol_size, enc_nf, dec_nf, src_feats, tgt_feats):
        super(UNet, self).__init__()
        self.encoders = nn.ModuleList()
        self.encoders_ = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.decoders_ = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.enc_nf = enc_nf
        self.dec_nf = dec_nf
        # Encoder path
        in_channels = src_feats + tgt_feats
        for out_channels in enc_nf:
            self.encoders.append(ConvBlock(in_channels, out_channels))  # Normal convolution
            self.encoders_.append(ConvBlock(in_channels=out_channels, out_channels=out_channels, stride=2))  # Downsampling convolution
            in_channels = out_channels

        # Decoder path
        self.decoders.append(ConvBlock(in_channels,dec_nf[0]))
        #in_channels +=dec_nf[0]
        """for out_channels in dec_nf[1:]:
            self.decoders.append(ConvBlock(in_channels, out_channels))
            self.decoders_.append(ConvBlock(out_channels, out_channels))
            self.upsample_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            in_channels = out_channels"""
        enc_nfnew = [src_feats + tgt_feats]+enc_nf
        for i in range(1, len(enc_nf)+1):
            #print(f"{i}: ", in_channels)
            in_channels += enc_nfnew[-(i+1)]
            self.upsample_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.decoders.append(ConvBlock(in_channels, dec_nf[i]))
            self.decoders_.append(ConvBlock(dec_nf[i], dec_nf[i]))
           # print("**")
            in_channels = dec_nf[i]
        
        print("final layers: ", dec_nf[len(enc_nf)],dec_nf[len(enc_nf)+1])
        self.final_conv1 = ConvBlock(in_channels, dec_nf[len(enc_nf)])
        self.final_conv2 = ConvBlock(dec_nf[len(enc_nf)] ,dec_nf[len(enc_nf)+1])
            
            
            

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        
        skips = [x]
        
        for i in range(0, len(self.encoders)):
           # print("skips being encoded: ", skips[-1].shape)
            x_temp = self.encoders[i](skips[-1])
          #  print("x_temp first encoding: ", x_temp.shape)
            skips.append(self.encoders_[i](x_temp))
            #x = 
            #print("x_temp second encoding: ", skips[-1].shape)
        #print("everything cool")
        # Bottleneck
        x = self.decoders[0](skips[-1])
        #print("here too: ", x.shape)
        # Decoder path
        rev_skip = skips[:-1][::-1]
        for decoder, decoder_, upsample, skip in zip(self.decoders[1:],self.decoders_, self.upsample_layers, rev_skip):
           # print("-")
          #  print("x being decoded: ", x.shape)
            x = upsample(x)
           # print("x after upsample: ", x.shape)
            x = torch.cat([x, skip], dim=1)
           # print("concat with skip: ", skip.shape)
            #print("x after cat: ", x.shape)
            x = decoder(x)
            x = decoder_(x)
            #print("after decode: ", x.shape)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        return x

    
class Aligner_unet_cvpr2018_vJX(nn.Module):
    def __init__(self, vol_size, enc_nf, dec_nf, indexing='ij', flow_only=False, gauss_kernal_size=79,
                              flow_clipping=False, flow_clipping_nsigma=1, flow_thresholding=False, flow_thresh_dis=30,
                              loss_mask=False, loss_mask_from_prev_cascade=False):
        
        super(Aligner_unet_cvpr2018_vJX, self).__init__()
        
        self.loss_mask = loss_mask
        self.loss_mask_from_prev_cascade = loss_mask_from_prev_cascade
        ndims = len(vol_size)
        self.unet_model = UNet(vol_size, enc_nf, dec_nf, src_feats=3, tgt_feats=3)
        if ndims ==2:
            self.conv = nn.Conv2d(dec_nf[-1], 2, kernel_size=3, padding = 1, bias = False)
            self.gauss = GaussianBlurLayer(2, kernel_size=gauss_kernal_size)
        if ndims ==3:
            self.conv = nn.Conv3d(dec_nf[-1], 3,  kernel_size=3, padding = 1, bias = False)    
            self.gauss = GaussianBlurLayer(3, kernel_size=gauss_kernal_size)
        nn.init.normal_(self.conv.weight, mean=0.0, std=1e-5)
        self.flow_clipping = flow_clipping
        self.flow_clipping_nsigma = flow_clipping_nsigma
        self.flow_thresholding = flow_thresholding
        self.flow_thresh_dis = flow_thresh_dis
        self.flow_only = flow_only
        self.indexing = indexing
        self.transformer =  SpatialTransformer(interp_method='linear', indexing=self.indexing)
    def forward(self, src, tgt):
        if self.loss_mask:
            if self.loss_mask_from_prev_cascade:
                # Assume the last channel in 'moving' is the mask
                tgt = src[:, :, :, :-1]  # Excludes the last channel
                train_loss_mask = src[:, :, :, -1:]  # Isolates the last channel as the mask
                x = self.unet_model(src, tgt)
            else:
                # Initialize a new mask with padding
                train_loss_mask = torch.ones((src.size(0), self.vol_size[0] - 4, self.vol_size[1] - 4, 1),
                                             device=src.device, dtype=torch.float32)
                # Apply padding
                paddings = (2, 2, 2, 2)  # Padding for the last two dimensions
                train_loss_mask = F.pad(train_loss_mask, paddings, "constant", 0)
        x = self.unet_model(src, tgt)
        flow = self.conv(x)
        #print("flow pre clip: ", flow)
        #print("flow:", flow.shape)
        flow_before_clipping = flow
        if self.flow_clipping:
            assert self.flow_clipping_nsigma is not None
            flow_mean = torch.mean(flow, dim=[2, 3], keepdim=True)
           # print("flow:", flow.shape)
           # print("flow mean: ", flow_mean.shape, flow_mean)
            flow_std = torch.std(flow, dim=[2, 3], keepdim=True, unbiased=False)
           # print("flow:", flow.shape)
            #print("flow std: ", flow_std.shape, flow_std)
            clip_min = (flow_mean - self.flow_clipping_nsigma * flow_std)
            #print("flow:", flow.shape)
            clip_max = (flow_mean + self.flow_clipping_nsigma * flow_std)
            #print("Clip: ", clip_min, clip_max)
           ## print("flow:", flow.shape)
            if self.flow_thresholding:
                flow_thresh_dis_tensor = torch.tensor(self.flow_thresh_dis, dtype=clip_min.dtype, device=clip_min.device)
                clip_min = torch.max(clip_min, flow_thresh_dis_tensor)
                clip_max = torch.min(clip_max, flow_thresh_dis_tensor)
            #print("Clip: ", clip_min, clip_max)
            flow = torch.clamp(flow, min=clip_min, max=clip_max)
            #print("flow:", flow.shape)
        elif self.flow_thresholding:
            flow = torch.clamp(flow, -self.flow_thersh_dis, self.flow_thersh_dis)
        #print("flow:", flow.shape)   
        flow = self.gauss(flow)
       #print("flow after gauss: ", flow.shape)
        if self.loss_mask:
            moving_for_warp = torch.cat([src, train_loss_mask], dim=1)
        else:
            moving_for_warp = src
            
        #print("moving_warp shape: ", moving_for_warp.shape, flow.shape)
        
        if not self.flow_only:
            

            moving_transformed, flow = self.transformer([moving_for_warp, flow]) # applies the dvf to the image 
            #print("mv", moving_transformed.shape)
            #moving_transformed = moving_transformed.permute(0,3,1,2)
            #print("flow dim: ", flow.shape)
            if self.flow_clipping or self.flow_thresholding:
                return moving_transformed, flow, flow_before_clipping
            else:
                return moving_transformed, flow
            
        else:
            return flow
                  
            
