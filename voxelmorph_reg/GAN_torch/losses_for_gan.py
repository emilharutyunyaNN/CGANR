import torch
import torch.nn.functional as F
import numpy as np
#
import lpips
from DISTS_pytorch import DISTS


# suppose we have a tensor and we want to get patches from it
def get_patches(img, patch_size):
    img = img.unsqueeze(0)
    img = torch.clamp(img, 0, 1)
    img = (img - 0.5) * 2
    #print("//: ", img.shape)
    #print("---", img.size(1))
    patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(-1, img.size(1), patch_size, patch_size)
    return patches

# Function to compute average LPIPS loss over patches for a single image
def compute_image_lpips_loss(img0, img1, patch_size, loss_fn):
    patches0 = get_patches(img0, patch_size)
    patches1 = get_patches(img1, patch_size)

    lpips_losses = []
    for i in range(patches0.size(0)):
        patch0 = patches0[i].unsqueeze(0)
        patch1 = patches1[i].unsqueeze(0)
        loss = loss_fn(patch0, patch1)
        #sprint("Loss lpips: ", loss)
        lpips_losses.append(loss.item())

    avg_loss = sum(lpips_losses) / len(lpips_losses)
    return avg_loss

# Function to compute the batch mean LPIPS loss
def compute_batch_lpips_loss(batch_img0, batch_img1, patch_size, model):
    """
    computes for a batch for each image calculates ths mean of distances of corresponding patches
    then sums them up with those of other batches and returns and average value
    """
    
    batch_size = batch_img0.size(0)
    image_losses = []
    #loss_fn = lpips.LPIPS(net=net, spatial=True).to(batch_img0.device)
    
    for b in range(batch_size):
        img0 = batch_img0[b]
        img1 = batch_img1[b]
        avg_image_loss = compute_image_lpips_loss(img0, img1, patch_size, model)
        image_losses.append(avg_image_loss)
    
    avg_batch_loss = sum(image_losses) / len(image_losses)
    return avg_batch_loss

#DISTS loss

def dist_train(img1, img2):
    """
        dists loss for training
    """
    D = DISTS().to(img1.device)
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    dists_loss = D(img1, img2, require_grad=True, batch_average=True)
    return dists_loss

def dist_test(img1, img2):
    """
        dists loss for testing
    """
    D = DISTS()
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    dists_loss = D(img1, img2)
    return dists_loss

def total_variation_loss(img):
    """
    Compute the Total Variation Loss for the batch.
    
    Parameters:
    img (torch.Tensor): The input tensor containing images. Shape [batch_size, channels, height, width]
    
    Returns:
    torch.Tensor: Total Variation Loss for the batch. Shape []
    """
    # Calculate the differences of neighboring pixel-values
    pixel_dif1 = img[:, :, 1:, :] - img[:, :, :-1, :]  # Difference across height
    pixel_dif2 = img[:, :, :, 1:] - img[:, :, :, :-1]  # Difference across width
    
    # Calculate the absolute differences and sum them over batch, channels, height, and width
    loss = torch.sum(torch.abs(pixel_dif1)) + torch.sum(torch.abs(pixel_dif2))
    
    return loss

class Grad:
    def __init__(self, penalty = 'l1', loss_mult = None, vox_weight = None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        self.vox_weight = vox_weight
        
    def _diffs(self, y):
        vol_shape = y.shape[1:-1]
        n_dims = len(vol_shape)
        df = [None] * n_dims
        for i in range(n_dims):
            d = i+1
            r = [d] + list(range(d)) + list(range(d + 1, n_dims + 2))
            yp = y.permute(r)
            dfi = yp[1:, ...] - yp[:-1, ...]
            
            if self.vox_weight is not None:
                w = self.vox_weight.permute(r)
                dfi = w[1:, ...]*dfi
            r = list(range(1, d + 1)) + [0] + list(range(d + 1, n_dims + 2))
            df[i] = dfi.permute(r)
        return df
    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]
        #print("dif: ", dif[0].shape)
        df = [torch.mean(f.reshape(f.size(0), -1), dim=1) for f in dif]
        grad = torch.sum(torch.stack(df), dim=0) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad

class ExponentialMovingAverage:
    def __init__(self, alpha = 0.1):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value == None:
            self.value = new_value
        else:
            self.value = self.alpha*new_value + (1-self.alpha)*self.value
        return self.value
    
class NCC:
    """
    Local (over window) normalized cross-correlation loss.
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps

    def ncc(self, Ii, Ji):
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats] in my case it is [B, nb_features, *vol_shape]
        ndims = Ii.dim() - 2
        #print("Ii:-------------------", Ii.dim())
        assert ndims in [1, 2, 3], f"volumes should be 1 to 3 dimensions. found: {ndims}"

        # set window size
        if self.win is None:
            self.win = [9] * ndims
        elif not isinstance(self.win, list):  # user specified a single number not a list
            self.win = [self.win] * ndims

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)
       # print(Ii.shape, Ji.shape)
        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji
       # print(I2.shape, J2.shape, IJ.shape)
        # compute filters
        in_ch = Ji.size(1)
        sum_filt = torch.ones([1, in_ch, *self.win]).to(Ii.device)
        strides = [1]*ndims
        #print(in_ch, sum_filt.shape, strides)
        #if ndims > 1:
          #  strides = [1] * (ndims + 2)

        # compute local sums via convolution
        #print("kernel: ", sum_filt, "dilation: ", 1)
        padding = 'same'
        I_sum = conv_fn(Ii, sum_filt, stride=strides, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=strides, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=strides, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=strides, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=strides, padding=padding)

        # compute cross-correlation
        win_size = np.prod(self.win) * in_ch
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = torch.maximum(cross, torch.tensor(self.eps).to(cross.device))
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = torch.maximum(I_var, torch.tensor(self.eps).to(I_var.device))
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = torch.maximum(J_var, torch.tensor(self.eps).to(J_var.device))

        cc = (cross / I_var) * (cross / J_var)

        # return mean cc for each entry in batch
        return torch.mean(cc.view(cc.size(0), -1), dim=-1)

    def loss(self, y_true, y_pred):
        return 1 - self.ncc(y_true, y_pred)
def loss_D(D_real_output, D_fake_output):
    D_fake_loss = torch.mean(D_fake_output ** 2)
    D_real_loss = torch.mean((1 - D_real_output) ** 2)
    D_total_loss = D_fake_loss + D_real_loss
    return D_total_loss, D_real_loss, D_fake_loss
def l1_loss(output, target):
    loss = torch.mean(torch.abs(output - target))
    return loss
def huber_reverse_loss(pred, label, delta=0.2, adaptive=True):
    diff = torch.abs(pred - label)
    if adaptive:
        delta = delta * torch.std(label)  # batch-adaptive
    loss = torch.mean(
        (diff <= delta).float() * diff +
        (diff > delta).float() * (diff**2 / 2 + delta**2 / 2) / delta
    )
    return loss

#def loss_G(D_fake_output, G_outputs, target_transformed, train_config, cur_epoch):

def loss_G(D_fake_output, G_output, target, train_config, cur_epoch=None, option = 'dists'):
    
    if train_config.is_training and train_config.case_filtering \
            and cur_epoch >= train_config.case_filtering_starting_epoch:
        assert cur_epoch is not None
        print("CASE FILTERING")
        if train_config.case_filtering_metric == 'ncc':
            min_ncc_threshold = train_config.case_filtering_cur_mean - \
                                train_config.case_filtering_nsigma * train_config.case_filtering_cur_stdev
            target_clipped = torch.clamp(target, 0, 1)
            G_output_clipped = torch.clamp(G_output, 0, 1)
            if train_config.case_filtering_x_subdivision == 1 and train_config.case_filtering_y_subdivision == 1:
                cur_ncc = NCC(win=20, eps=1e-3) 
                with torch.no_grad():
                    cur_ncc = cur_ncc.ncc(target_clipped, G_output_clipped).detach() 
                cur_mask = (cur_ncc > min_ncc_threshold).int()
                train_config.epoch_filtering_ratio.append(1 - cur_mask.float().mean().item())
                
                cur_index = torch.nonzero(cur_mask, as_tuple=False).squeeze()

# Gather elements along the 0th dimension using the indices
                G_output = torch.index_select(G_output, 0, cur_index)
                target = torch.index_select(target, 0, cur_index)
                D_fake_output = torch.index_select(D_fake_output, 0, cur_index)

                if 'loss_mask' in train_config and train_config['loss_mask']:
                    loss_mask_from_R = torch.index_select(loss_mask_from_R, 0, cur_index)
                    
        
                
    G_berhu_loss = huber_reverse_loss(pred=G_output, label=target) # l1 loss
    G_tv_loss = torch.mean(total_variation_loss(G_output))/(train_config.image_size **2)
    G_dis_loss = torch.mean((1-D_fake_output)**2)
    #print("------------- ", G_berhu_loss, G_tv_loss, G_dis_loss)
    G_total_loss = G_berhu_loss +0.02* G_tv_loss + train_config.lamda * G_dis_loss

    return G_total_loss, G_dis_loss, G_berhu_loss


def loss_G_new(D_fake_output, G_output, target, train_config, cur_epoch=None, ema = None):
    
    if train_config.is_training and train_config.case_filtering \
            and cur_epoch >= train_config.case_filtering_starting_epoch:
        assert cur_epoch is not None
        print("CASE FILTERING")
        if train_config.case_filtering_metric == 'ncc':
            min_ncc_threshold = train_config.case_filtering_cur_mean - \
                                train_config.case_filtering_nsigma * train_config.case_filtering_cur_stdev
            target_clipped = torch.clamp(target, 0, 1)
            G_output_clipped = torch.clamp(G_output, 0, 1)
            if train_config.case_filtering_x_subdivision == 1 and train_config.case_filtering_y_subdivision == 1:
                cur_ncc = NCC(win=20, eps=1e-3) 
                with torch.no_grad():
                    cur_ncc = cur_ncc.ncc(target_clipped, G_output_clipped).detach() 
                cur_mask = (cur_ncc > min_ncc_threshold).int()
                train_config.epoch_filtering_ratio.append(1 - cur_mask.float().mean().item())
                
                cur_index = torch.nonzero(cur_mask, as_tuple=False).squeeze()

# Gather elements along the 0th dimension using the indices
                G_output = torch.index_select(G_output, 0, cur_index)
                target = torch.index_select(target, 0, cur_index)
                D_fake_output = torch.index_select(D_fake_output, 0, cur_index)

                if 'loss_mask' in train_config and train_config['loss_mask']:
                    loss_mask_from_R = torch.index_select(loss_mask_from_R, 0, cur_index)
                    
        
    if cur_epoch >1:            
        G_berhu_loss = huber_reverse_loss(pred=G_output, label=target) # l1 loss
    else:
        ncc = NCC(win=20, eps=1e-3) 
        with torch.no_grad():
            G_berhu_loss = ncc.ncc(target_clipped, G_output_clipped).detach() 
        G_berhu_loss = ema.update(G_berhu_loss)
        
    G_tv_loss = torch.mean(total_variation_loss(G_output))/(train_config.image_size **2)
    G_dis_loss = torch.mean((1-D_fake_output)**2)
    #print("------------- ", G_berhu_loss, G_tv_loss, G_dis_loss)
    G_total_loss = G_berhu_loss +0.02* G_tv_loss + train_config.lamda * G_dis_loss

    return G_total_loss, G_dis_loss, G_berhu_loss

def loss_G_perceptual(D_fake_output, G_output, target, train_config, model_perc,cur_epoch=None, option = 'lpips'):
    
    if train_config.is_training and train_config.case_filtering \
            and cur_epoch >= train_config.case_filtering_starting_epoch:
        assert cur_epoch is not None
        print("CASE FILTERING")
        if train_config.case_filtering_metric == 'ncc':
            min_ncc_threshold = train_config.case_filtering_cur_mean - \
                                train_config.case_filtering_nsigma * train_config.case_filtering_cur_stdev
            target_clipped = torch.clamp(target, 0, 1)
            G_output_clipped = torch.clamp(G_output, 0, 1)
            if train_config.case_filtering_x_subdivision == 1 and train_config.case_filtering_y_subdivision == 1:
                cur_ncc = NCC(win=20, eps=1e-3) 
                with torch.no_grad():
                    cur_ncc = cur_ncc.ncc(target_clipped, G_output_clipped).detach() 
                cur_mask = (cur_ncc > min_ncc_threshold).int()
                train_config.epoch_filtering_ratio.append(1 - cur_mask.float().mean().item())
                
                cur_index = torch.nonzero(cur_mask, as_tuple=False).squeeze()

# Gather elements along the 0th dimension using the indices
                G_output = torch.index_select(G_output, 0, cur_index)
                target = torch.index_select(target, 0, cur_index)
                D_fake_output = torch.index_select(D_fake_output, 0, cur_index)

                if 'loss_mask' in train_config and train_config['loss_mask']:
                    loss_mask_from_R = torch.index_select(loss_mask_from_R, 0, cur_index)
                    
   # print("G: ", G_output.shape)
    if cur_epoch ==0:            
        print("perceptual")
        if option == 'dists':
            G_berhu_loss = dist_train(G_output, target)
        elif option == 'lpips':
            G_berhu_loss = compute_batch_lpips_loss(G_output, target, 64, model_perc)
    else:
        G_berhu_loss = huber_reverse_loss(pred=G_output, label=target) # l1 loss
        
    G_tv_loss = torch.mean(total_variation_loss(G_output))/(train_config.image_size **2)
    G_dis_loss = torch.mean((1-D_fake_output)**2)
    #print("------------- ", G_berhu_loss, G_tv_loss, G_dis_loss)
    G_total_loss = G_berhu_loss +0.02* G_tv_loss + train_config.lamda * G_dis_loss

    return G_total_loss, G_dis_loss, G_berhu_loss



def loss_R_no_gt(R_outputs, fixed, training_config):
    if training_config.dvf_thresholding or training_config.dvf_clipping:
        moving_transformed, flow_pred, _ = R_outputs
    else:
        moving_transformed, flow_pred = R_outputs
    
    if training_config.R_loss_type == 'berhu':
        R_structure_loss = huber_reverse_loss(moving_transformed, fixed)
        
    else:
        assert training_config.R_loss_type == 'ncc'
        moving_transformed_clipped = torch.clamp(moving_transformed, 0,1)
        fixed_clipped = torch.clamp(fixed, 0, 1)
        ncc = NCC(win = 20, eps = 1e-3)
        print(fixed_clipped.shape, moving_transformed_clipped.shape)
        R_structure_loss = torch.mean(ncc.loss(y_true=fixed_clipped, y_pred = moving_transformed_clipped))
        
        
    R_flow_tv_loss = 0
    if training_config.lambda_r_tv>0:
        grad = Grad('l2')
        #print("flow: ", flow_pred.shape)
        flow_p = flow_pred.permute(0,2,3,1)
        R_flow_tv_loss = torch.mean(grad.loss(None, flow_p))
        
    R_total_loss = R_structure_loss+ training_config.lambda_r_tv*R_flow_tv_loss
    return R_total_loss, R_structure_loss

def loss_R_no_gt_perceptual(R_outputs, fixed, training_config,curr_epoch, model):
    if training_config.dvf_thresholding or training_config.dvf_clipping:
        moving_transformed, flow_pred, _ = R_outputs
    else:
        moving_transformed, flow_pred = R_outputs
    
    if curr_epoch>0:
        if training_config.R_loss_type == 'berhu':
            R_structure_loss = huber_reverse_loss(moving_transformed, fixed)
            
        else:
            assert training_config.R_loss_type == 'ncc'
            moving_transformed_clipped = torch.clamp(moving_transformed, 0,1)
            fixed_clipped = torch.clamp(fixed, 0, 1)
            ncc = NCC(win = 20, eps = 1e-3)
            print(fixed_clipped.shape, moving_transformed_clipped.shape)
            R_structure_loss = torch.mean(ncc.loss(y_true=fixed_clipped, y_pred = moving_transformed_clipped))
    else:
        R_structure_loss = compute_batch_lpips_loss(R_outputs, fixed, 64, model)
        
        
    R_flow_tv_loss = 0
    if training_config.lambda_r_tv>0:
        grad = Grad('l2')
        #print("flow: ", flow_pred.shape)
        flow_p = flow_pred.permute(0,2,3,1)
        R_flow_tv_loss = torch.mean(grad.loss(None, flow_p))
        
    R_total_loss = R_structure_loss+ training_config.lambda_r_tv*R_flow_tv_loss
    return R_total_loss, R_structure_loss
        
def split_tensor(inp, x_split_times, y_split_times):
    if x_split_times > 1:
        x_splitted = torch.split(inp, inp.shape[2] // x_split_times, dim=2)
        x_splitted_concat = torch.cat(x_splitted, dim=0)
    else:
        x_splitted_concat = inp
    
    if y_split_times > 1:
        y_splitted = torch.split(x_splitted_concat, x_splitted_concat.shape[3] // y_split_times, dim=3)
        y_splitted_concat = torch.cat(y_splitted, dim=0)
    else:
        y_splitted_concat = x_splitted_concat
    
    return y_splitted_concat    
import pytorch_ssim as ps
def compute_ssim(img1, img2, max_val = 1):
    
    return ssim(img1, img2, data_range=max_val)
def compute_psnr(img1, img2, max_val=1.0):
    """
    Compute the Peak Signal-to-Noise Ratio between two images in PyTorch.

    Args:
        img1 (torch.Tensor): First image tensor, shape [N, C, H, W].
        img2 (torch.Tensor): Second image tensor, shape [N, C, H, W].
        max_val (float): Maximum possible pixel value of the images.

    Returns:
        torch.Tensor: PSNR values of each image in the batch, shape [N].
    """
    # Ensure the input tensors are of type float32
    img1 = img1.type(torch.float32)
    img2 = img2.type(torch.float32)

    # Scale max_val to float32 if it is not already a float
    max_val = float(max_val)

    # Calculate MSE between the two images
    mse = F.mse_loss(img1, img2, reduction='none')
    mse = mse.mean([1, 2, 3])  # Mean over the channel, height, and width

    # Avoid division by zero
    mse = mse.clamp(min=1e-10)

    # Calculate PSNR
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))

    return psnr

import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input: Tensor, win: Tensor) -> Tensor:
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float,
    win: Tensor,
    size_average: bool = True,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
) -> Tuple[Tensor, Tensor]:
    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        data_range (float or int): value range of input images. (usually 1.0 or 255)
        win (torch.Tensor): 1-D gauss kernel
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    nonnegative_ssim: bool = False,
) -> Tensor:
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    X: Tensor,
    Y: Tensor,
    data_range: float = 255,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    win: Optional[Tensor] = None,
    weights: Optional[List[float]] = None,
    K: Union[Tuple[float, float], List[float]] = (0.01, 0.03)
) -> Tensor:
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    #if not X.type() == Y.type():
    #    raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights_tensor = X.new_tensor(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights_tensor.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # type: ignore  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
        nonnegative_ssim: bool = False,
    ) -> None:
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range: float = 255,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        channel: int = 3,
        spatial_dims: int = 2,
        weights: Optional[List[float]] = None,
        K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
    ) -> None:
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )
        
        
#loss for affine
import torch
import torch.nn.functional as F

def affine_transformation_loss(output, target, transformation_matrix, lambda_smooth=0.01, lambda_trans=0.001):
    # Content loss (MSE or L1)
   # print("Output: ", output)
    content_loss = F.mse_loss(output, target)
    
    # Identity matrix for comparison
    identity_matrix = torch.eye(2, 3).unsqueeze(0).to(transformation_matrix.device)
    if transformation_matrix.size(0) != 1:
        identity_matrix = identity_matrix.repeat(transformation_matrix.size(0), 1, 1)
    
    # Regularization for non-translation components
    smoothness_loss = F.mse_loss(transformation_matrix[:, :, :2], identity_matrix[:, :, :2])
    
    # Scaled translation penalty to allow for more significant translation
    translation_penalty = lambda_trans * torch.sum(transformation_matrix[:, :, 2] ** 2)
    
    # Total loss
    total_loss = content_loss + lambda_smooth * smoothness_loss + translation_penalty
    return total_loss