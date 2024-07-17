import torch
import numpy as np


def spatial_transform_network(input_fmap, theta, out_dims = None, **kwargs):
    """
    Spatial Transformer Network layer implementation as described in [1].
    The layer is composed of 3 elements:
    - localization_net: takes the original image as input and outputs
      the parameters of the affine transformation that should be applied
      to the input image.
    - affine_grid_generator: generates a grid of (x,y) coordinates that
      correspond to a set of points where the input should be sampled
      to produce the transformed output.
    - bilinear_sampler: takes as input the original image and the grid
      and produces the output image using bilinear interpolation.
    Input
    -----
    - input_fmap: output of the previous layer. Can be input if spatial
      transformer layer is at the beginning of architecture. Should be
      a tensor of shape (B, H, W, C).
    - theta: affine transform tensor of shape (B, 6). Permits cropping,
      translation and isotropic scaling. Initialize to identity matrix.
      It is the output of the localization network.
    Returns
    -------
    - out_fmap: transformed input feature map. Tensor of size (B, H, W, C).
    Notes
    -----
    [1]: 'Spatial Transformer Networks', Jaderberg et. al,
         (https://arxiv.org/abs/1506.02025)
    """
    
    B,H,W,C = input_fmap.shape
    #reshape theta to (B,2,3)
    theta = theta.view(B,2,3)
    print(theta)
    if out_dims:
        out_H = out_dims[1]
        out_W = out_dims[2]
        batch_grids = affine_grid_generator(out_H, out_W, theta)
    else:
        batch_grids = affine_grid_generator(H, W, theta)
    
    print("batch grid:", batch_grids)
    x_s = batch_grids[:,0,:,:]
    y_s = batch_grids[:,1,:,:]
    print("xsys: ", x_s,y_s)
    print(input_fmap.shape, x_s.shape, y_s.shape)
    out_fmap = bilinear_sampler(input_fmap, x_s,y_s)
    print("out fmap: ", out_fmap.shape)
    return out_fmap

# theta is a tensor that is not a single theta it can be in batches
def affine_grid_generator(height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation of the input feature map.
    
    Input
    -----
    - height: desired height of grid/output. Used to downsample or upsample.
    - width: desired width of grid/output. Used to downsample or upsample.
    - theta: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 theta parameters of
      the form (2x3) that define the affine transformation T.
    
    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.
    """
    num_batch = theta.shape[0]
    
    
    # sampling grid doesn't change when theta is identity transforms
    
    
    # Create normalized 2D grid
    x = torch.linspace(-1.0, 1.0, steps=width)
    y = torch.linspace(-1.0, 1.0, steps=height)
    #print("xy: ", x,y)
    y_t, x_t = torch.meshgrid(y, x)  # Note the order: y first, x second
    #print("yt: ",y_t, "\n", "xt: ", x_t)
    # Flatten
    x_t_flat = x_t.reshape(-1)
    y_t_flat = y_t.reshape(-1)
    # print("flattened: ", x_t_flat, y_t_flat)
    # Reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = torch.ones_like(x_t_flat)
    sampling_grid = torch.stack([x_t_flat, y_t_flat, ones])  # Shape: [3, H * W]
    #print(sampling_grid.shape)
    # Repeat grid num_batch times
    sampling_grid = sampling_grid.unsqueeze(0)  # Shape: [1, 3, H * W]
    sampling_grid = sampling_grid.repeat(num_batch, 1, 1)  # Shape: [num_batch, 3, H * W]

    # Cast to float32 (required for matmul)
    theta = theta.float()
    sampling_grid = sampling_grid.float()

    # Transform the sampling grid - batch multiply
    batch_grids = torch.bmm(theta, sampling_grid)  # Shape: [num_batch, 2, H * W]
    #print(batch_grids.shape)
    # Reshape to (num_batch, 2, H, W)
    batch_grids = batch_grids.view(num_batch, 2, height, width) # view is important
    #print("affine grid: ", batch_grids)
    #print(sampling_grid, batch_grids)
    return batch_grids

"""def get_pixel_value(img, x, y):
    
    shape = x.shape
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = torch.arange(0, batch_size).view(batch_size, 1, 1).to(img.device)
    b = batch_idx.repeat(1, height, width)

    # Stack to create indices of shape (B, H, W, 3)
    indices = torch.stack([b, y, x], dim=3)

    # Use gather to get the pixel values from img
    output = img[indices[..., 0], indices[..., 1], indices[..., 2]]
    print("get_pixel: ",output.shape)
    return output"""


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a 4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: tensor of shape (B, H, W)
    - y: tensor of shape (B, H, W)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    batch_size, height, width, channels = img.shape
    batch_idx = torch.arange(0, batch_size).view(batch_size, 1, 1).to(img.device)
    b = batch_idx.repeat(1, height, width)

    indices = torch.stack([b, y, x], dim=3)

    return img[indices[..., 0], indices[..., 1], indices[..., 2]]

def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - x, y: normalized coordinates.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H, W = img.shape[1:3]
    max_y = H - 1
    max_x = W - 1
    zero = 0

    # rescale x and y to [0, W-1] / [0, H-1]
    x = 0.5 * ((x + 1.0) * (max_x ))
    y = 0.5 * ((y + 1.0) * (max_y ))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = torch.floor(x).to(torch.int32)
    x1 = x0 + 1
    y0 = torch.floor(y).to(torch.int32)
    y1 = y0 + 1

    # clip to range [0, H-1] / [0, W-1] to not violate img boundaries
    x0_ = torch.clamp(x0, zero, max_x)
    x1_ = torch.clamp(x1, zero, max_x)
    y0_ = torch.clamp(y0, zero, max_y)
    y1_ = torch.clamp(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0_, y0_)
    Ib = get_pixel_value(img, x0_, y1_)
    Ic = get_pixel_value(img, x1_, y0_)
    Id = get_pixel_value(img, x1_, y1_)

    # recast as float for delta calculation
    x0 = x0.to(torch.float32)
    x1 = x1.to(torch.float32)
    y0 = y0.to(torch.float32)
    y1 = y1.to(torch.float32)

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    
    print("ws: ", wa, wb, wc, wd)

    # add dimension for addition
    wa = wa.unsqueeze(3)
    wb = wb.unsqueeze(3)
    wc = wc.unsqueeze(3)
    wd = wd.unsqueeze(3)

    # compute output
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return out

# Example usage:
# img is a tensor of shape (B, H, W, C)
# x and y are the outputs of affine_grid_generator
#-----------works
"""B, H, W, C = 2, 5, 5, 3
img = torch.rand(B, H, W, C)
print(img)
x = torch.linspace(-1, 1, steps=W).unsqueeze(0).repeat(H, 1).unsqueeze(0).repeat(B, 1, 1)
y = torch.linspace(-1, 1, steps=H).unsqueeze(1).repeat(1, W).unsqueeze(0).repeat(B, 1, 1)
output = bilinear_sampler(img, x, y)
print(output)
    
print((img==output))   """ 











import torch
import torch.nn.functional as F

def get_pixel_value(img, x, y):
    
    batch_size, height, width, channels = img.shape
    batch_idx = torch.arange(0, batch_size).view(batch_size, 1, 1).to(img.device)
    b = batch_idx.repeat(1, height, width)

    indices = torch.stack([b, y, x], dim=3)
    print("getting: ", indices, indices[..., 0],indices[..., 1],indices[..., 2])
    return img[indices[..., 0], indices[..., 1], indices[..., 2]]

def bilinear_sampler(img, x, y):
    
    B, H, W, C = img.shape
    max_y = H - 1
    max_x = W - 1
    zero = 0

    # Rescale x and y to [0, W-1] / [0, H-1]
    x = 0.5 * ((x + 1.0) * (max_x-1))
    y = 0.5 * ((y + 1.0) * (max_y-1))
    print(x,y)
    # Grab 4 nearest corner points for each (x_i, y_i)
    x0 = torch.floor(x).to(torch.long)
    x1 = x0 + 1
    y0 = torch.floor(y).to(torch.long)
    y1 = y0 + 1
    print("x0: ", x0 )
    print("x1: ", x1 )
    print("y0: ", y0 )
    print("y1: ", y1 )

    # Clip to range [0, H-1] / [0, W-1] to not violate img boundaries
    x0 = torch.clamp(x0, zero, max_x)
    x1 = torch.clamp(x1, zero, max_x)
    y0 = torch.clamp(y0, zero, max_y)
    y1 = torch.clamp(y1, zero, max_y)

    # Get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    print("Ia:", Ia)
    Ib = get_pixel_value(img, x0, y1)
    print("Ib: ", Ib)
    Ic = get_pixel_value(img, x1, y0)
    print("Ic: ", Ic)
    Id = get_pixel_value(img, x1, y1)
    print("Id: ", Id)
    # Recast as float for delta calculation
    x0 = x0.to(torch.float32)
    x1 = x1.to(torch.float32)
    y0 = y0.to(torch.float32)
    y1 = y1.to(torch.float32)

    # Calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # Add dimension for addition
    wa = wa.unsqueeze(3)
    wb = wb.unsqueeze(3)
    wc = wc.unsqueeze(3)
    wd = wd.unsqueeze(3)
    
    print("wa: ", wa)
    print("wb: ", wb)
    print("wc: ", wc)
    print("wd: ", wd)

    # Compute output
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return out

def affine_grid_generator(height, width, theta):
    
    num_batch = theta.shape[0]

    # Create normalized 2D grid
    x = torch.linspace(-1.0, 1.0, steps=width)
    y = torch.linspace(-1.0, 1.0, steps=height)
    y_t, x_t = torch.meshgrid(y, x)  # Note the order: y first, x second

    # Flatten
    x_t_flat = x_t.reshape(-1)
    y_t_flat = y_t.reshape(-1)

    # Reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = torch.ones_like(x_t_flat)
    sampling_grid = torch.stack([x_t_flat, y_t_flat, ones])  # Shape: [3, H * W]

    # Repeat grid num_batch times
    sampling_grid = sampling_grid.unsqueeze(0).repeat(num_batch, 1, 1)  # Shape: [num_batch, 3, H * W]

    # Cast to float32 (required for matmul)
    theta = theta.float()
    sampling_grid = sampling_grid.float()

    # Transform the sampling grid - batch multiply
    batch_grids = torch.bmm(theta, sampling_grid)  # Shape: [num_batch, 2, H * W]

    # Reshape to (num_batch, 2, H, W)
    batch_grids = batch_grids.view(num_batch, 2, height, width)

    return batch_grids

def spatial_transform_network(input_fmap, theta, out_dims=None):
    
    B, H, W, C = input_fmap.shape
    # Reshape theta to (B, 2, 3)
    theta = theta.view(B, 2, 3)

    if out_dims:
        out_H = out_dims[1]
        out_W = out_dims[2]
        batch_grids = affine_grid_generator(out_H, out_W, theta)
    else:
        batch_grids = affine_grid_generator(H, W, theta)

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]
    print("xsys: ", x_s,y_s)
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)
    return out_fmap

# Example usage:
"""B, H, W, C = 1, 5, 5, 3
img = img = torch.rand(B, H, W, C)
                 

# Identity affine matrix
theta = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32)

output = spatial_transform_network(img, theta)

# Check if output is the same as input
print("Input:")
print(img)
print("Output:")
print(output)
print("Are input and output identical?", torch.allclose(img, output, atol=1e-6))"""

"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
# Load CIFAR-10 dataset
import cv2
img = cv2.imread("./voxelmorph_reg/aligners/stained_patch_1_16384_33792_12.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Convert to PyTorch tensor and add batch dimension
img = transforms.ToTensor()(img).unsqueeze(0)

# Identity affine matrix
theta = torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float32)

output = spatial_transform_network(img, theta)

# Convert back to (B, C, H, W) for visualization
img = img.permute(0, 2, 3, 1)
output = output.permute(0, 2, 3, 1)

# Plot the input and output images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img[0].cpu().numpy())
axes[0].set_title('Input Image')
axes[0].axis('off')

axes[1].imshow(output[0].cpu().numpy())
axes[1].set_title('Output Image')
axes[1].axis('off')

plt.savefig("./voxelmorph_reg/aligners/align_test.jpg")

print("Are input and output identical?", torch.allclose(img, output, atol=1e-6))"""
"""
print("Input:")
print(img)
print("Output:")
print(output)
print("Are input and output identical?", torch.allclose(img, output, atol=1e-6))"""

def affine_transform(img, theta, size=None):
    """
    Apply affine transformation directly to image.
    img: Input image tensor of shape (B, C, H, W).
    theta: Affine transform parameters tensor of shape (B, 2, 3).
    size: Output size of the transformed image (H, W). If None, same size as input.
    Returns transformed image tensor.
    """
    B, C, H, W = img.shape
    #ssprint("affine input: ", B, C, H, W)

    # Generate grid of coordinates
    if size is None:
        size = (H, W)
    grid = F.affine_grid(theta, [B, C, *size], align_corners=False)

    # Apply affine transformation to image
    transformed_img = F.grid_sample(img, grid, align_corners=False)

    return transformed_img
"""  
output = affine_transform(img, theta)

print("Input:")
print(img)
print("Output:")
print(output)
print("Are input and output identical?", torch.allclose(img, output, atol=1e-6))"""

