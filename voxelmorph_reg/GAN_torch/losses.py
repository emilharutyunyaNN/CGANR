import torch
import torch.nn.functional as F

import numpy as np
import torch

import torch
import numpy as np

def _crps_torch(y_true, y_pred, factor=0.05, epsilon=1e-6):
    '''
    Core of (pseudo) CRPS loss.
    
    y_true: two-dimensional arrays or tensors
    y_pred: two-dimensional arrays or tensor
    factor: impore of std term
    epsilon: a small value to ensure numerical stability
    '''
    # Convert to torch tensors if inputs are numpy arrays
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    # mean absolute error
    mae = torch.mean(torch.abs(y_pred - y_true))
    
    # standard deviation within each batch
    dist = torch.std(y_pred) + epsilon
    
    return mae - factor * dist


def crps2d_torch(y_true, y_pred, factor=0.05, epsilon=1e-6):
    '''
    (Experimental)
    An approximated continuous ranked probability score (CRPS) loss function:
    
        CRPS = mean_abs_err - factor * std
        
    * Note that the "real CRPS" = mean_abs_err - mean_pairwise_abs_diff
    
    Replacing mean pairwise absolute difference by standard deviation offers
    a complexity reduction from O(N^2) to O(N*logN) 
    
    ** factor > 0.1 may yield negative loss values.
    
    Compatible with high-level PyTorch training methods
    
    Input
    ----------
        y_true: training target with shape=(batch_num, x, y, 1)
        y_pred: a forward pass with shape=(batch_num, x, y, 1)
        factor: relative importance of standard deviation term.
        epsilon: a small value to ensure numerical stability
    '''
    
    # Convert to torch tensors if inputs are numpy arrays
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    y_pred = torch.squeeze(y_pred, dim=-1)
    y_true = torch.squeeze(y_true, dim=-1)
    
    batch_num = y_pred.shape[0]
    
    crps_out = 0
    for i in range(batch_num):
        crps_out += _crps_torch(y_true[i, ...], y_pred[i, ...], factor=factor, epsilon=epsilon)
        
    return crps_out / batch_num

# Example data for testing with batch dimension and singleton dimension at the end
y_true_example = np.array([[[[1.0], [2.0], [3.0], [4.0]]]])
y_pred_example = np.array([[[[1.2], [1.9], [2.8], [3.9]]]])

# Testing _crps_torch with numpy arrays
print("Testing _crps_torch with numpy arrays:")
print(_crps_torch(y_true_example[0], y_pred_example[0], factor=0.05).item())

# Testing crps2d_torch with numpy arrays
print("\nTesting crps2d_torch with numpy arrays:")
print(crps2d_torch(y_true_example, y_pred_example, factor=0.05).item())

y_true_example = np.array([[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]])
y_pred_example = np.array([[[[1.2]], [[1.9]], [[2.8]], [[3.9]]]])
print(y_true_example.shape)
print(y_pred_example.shape)
# Testing _crps_torch with numpy arrays
print("Testing _crps_torch with numpy arrays:")
print(_crps_torch(y_true_example[0], y_pred_example[0], factor=0.05))

# Testing crps2d_torch with numpy arrays
print("\nTesting crps2d_torch with numpy arrays:")
print(crps2d_torch(y_true_example, y_pred_example, factor=0.05))


def _crps_np(y_true, y_pred, factor=0.05):
    
    '''
    Numpy version of _crps_tf.
    '''
    
    # mean absolute error
    mae = np.nanmean(np.abs(y_pred - y_true))
    dist = np.nanstd(y_pred)
    
    return mae - factor*dist

def crps2d_np(y_true, y_pred, factor=0.05):
    
    '''
    (Experimental)
    Nunpy version of `crps2d_tf`.
    
    Documentation refers to `crps2d_tf`.
    '''
    
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    
    batch_num = len(y_pred)
    
    crps_out = 0
    for i in range(batch_num):
        crps_out += _crps_np(y_true[i, ...], y_pred[i, ...], factor=factor)
        
    return crps_out/batch_num

y_true_example = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]], [[[1.5], [2.5]], [[3.5], [4.5]]]])
y_pred_example = np.array([[[[1.2], [1.9]], [[2.8], [3.9]]], [[[1.4], [2.6]], [[3.6], [4.4]]]])

# Testing _crps_np with numpy arrays
print("Testing _crps_np with numpy arrays:")
print(_crps_np(y_true_example[0], y_pred_example[0], factor=0.05))

# Testing crps2d_np with numpy arrays
print("\nTesting crps2d_np with numpy arrays:")
print(crps2d_np(y_true_example, y_pred_example, factor=0.05))


import torch
import numpy as np

def dice_coef(y_true, y_pred, const=1e-7):
    '''
    Sørensen–Dice coefficient for 2-d samples.
    
    Input
    ----------
        y_true, y_pred: predicted outputs and targets.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    # Convert numpy arrays to tensors if needed
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    # flatten 2-d tensors
    y_true_pos = y_true.view(-1)
    y_pred_pos = y_pred.view(-1)
    
    # get true pos (TP), false neg (FN), false pos (FP)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
    
    # 2TP/(2TP+FP+FN)
    coef_val = (2.0 * true_pos + const) / (2.0 * true_pos + false_pos + false_neg)
    
    return coef_val

def dice(y_true, y_pred, const=1e-7):
    '''
    Sørensen–Dice Loss.
    
    dice(y_true, y_pred, const=1e-7)
    
    Input
    ----------
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    # Convert numpy arrays to tensors if needed
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    # Squeeze-out length-1 dimensions
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    
    loss_val = 1 - dice_coef(y_true, y_pred, const=const)
    
    return loss_val


# Example data for testing
y_true_np = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
y_pred_np = np.array([[0.8, 0.2, 0.6], [0.1, 0.9, 0.4]], dtype=np.float32)

y_true_torch = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)
y_pred_torch = torch.tensor([[0.8, 0.2, 0.6], [0.1, 0.9, 0.4]], dtype=torch.float32)

# Testing dice_coef with numpy arrays
print("Testing dice_coef with numpy arrays:")
print(dice_coef(y_true_np, y_pred_np))

# Testing dice with numpy arrays
print("\nTesting dice with numpy arrays:")
print(dice(y_true_np, y_pred_np))

# Testing dice_coef with PyTorch tensors
print("\nTesting dice_coef with PyTorch tensors:")
print(dice_coef(y_true_torch, y_pred_torch))

# Testing dice with PyTorch tensors
print("\nTesting dice with PyTorch tensors:")
print(dice(y_true_torch, y_pred_torch))


import torch
import numpy as np

def tversky_coef(y_true, y_pred, alpha=0.5, const=1e-7):
    '''
    Weighted Sørensen–Dice coefficient.
    
    Input
    ----------
        y_true, y_pred: predicted outputs and targets.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    # Convert numpy arrays to tensors if needed
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    # flatten 2-d tensors
    y_true_pos = y_true.view(-1)
    y_pred_pos = y_pred.view(-1)
    
    # get true pos (TP), false neg (FN), false pos (FP)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
    
    # TP/(TP + a*FN + b*FP); a+b = 1
    coef_val = (true_pos + const) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + const)
    
    return coef_val

def tversky(y_true, y_pred, alpha=0.5, const=1e-7):
    '''
    Tversky Loss.
    
    tversky(y_true, y_pred, alpha=0.5, const=1e-7)
    
    ----------
    Hashemi, S.R., Salehi, S.S.M., Erdogmus, D., Prabhu, S.P., Warfield, S.K. and Gholipour, A., 2018. 
    Tversky as a loss function for highly unbalanced image segmentation using 3d fully convolutional deep networks. 
    arXiv preprint arXiv:1803.11078.
    
    Input
    ----------
        alpha: tunable parameter within [0, 1]. Alpha handles imbalance classification cases.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    # Convert numpy arrays to tensors if needed
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    # Squeeze-out length-1 dimensions
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    
    loss_val = 1 - tversky_coef(y_true, y_pred, alpha=alpha, const=const)
    
    return loss_val


def focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3, const=1e-7):
    '''
    Focal Tversky Loss (FTL)
    
    focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)
    
    ----------
    Abraham, N. and Khan, N.M., 2019, April. A novel focal tversky loss function with improved 
    attention u-net for lesion segmentation. In 2019 IEEE 16th International Symposium on Biomedical Imaging 
    (ISBI 2019) (pp. 683-687). IEEE.
    
    ----------
    Input
        alpha: tunable parameter within [0, 1]. Alpha handles imbalance classification cases 
        gamma: tunable parameter within [1, 3].
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    # Convert numpy arrays to tensors if needed
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    # Squeeze-out length-1 dimensions
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    
    # (Tversky loss)**(1/gamma)
    loss_val = torch.pow((1 - tversky_coef(y_true, y_pred, alpha=alpha, const=const)), 1/gamma)
    
    return loss_val


# Example data for testing
y_true_np = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
y_pred_np = np.array([[0.8, 0.2, 0.6], [0.1, 0.9, 0.4]], dtype=np.float32)

y_true_torch = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)
y_pred_torch = torch.tensor([[0.8, 0.2, 0.6], [0.1, 0.9, 0.4]], dtype=torch.float32)

# Testing tversky_coef with numpy arrays
print("Testing tversky_coef with numpy arrays:")
print(tversky_coef(y_true_np, y_pred_np))

# Testing tversky with numpy arrays
print("\nTesting tversky with numpy arrays:")
print(tversky(y_true_np, y_pred_np))

# Testing focal_tversky with numpy arrays
print("\nTesting focal_tversky with numpy arrays:")
print(focal_tversky(y_true_np, y_pred_np))

# Testing tversky_coef with PyTorch tensors
print("\nTesting tversky_coef with PyTorch tensors:")
print(tversky_coef(y_true_torch, y_pred_torch))

# Testing tversky with PyTorch tensors
print("\nTesting tversky with PyTorch tensors:")
print(tversky(y_true_torch, y_pred_torch))

# Testing focal_tversky with PyTorch tensors
print("\nTesting focal_tversky with PyTorch tensors:")
print(focal_tversky(y_true_torch, y_pred_torch))


from pytorch_msssim import ms_ssim

def ms_ssim_loss(y_true, y_pred, **kwargs):
    """
    Multiscale structural similarity (MS-SSIM) loss for PyTorch.
    
    ms_ssim_loss(y_true, y_pred, **kwargs)
    
    ----------
    Wang, Z., Simoncelli, E.P. and Bovik, A.C., 2003, November. Multiscale structural similarity for image quality assessment. 
    In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003 (Vol. 2, pp. 1398-1402). Ieee.
    
    ----------
    Input
        kwargs: keywords of `pytorch_msssim.ms_ssim`
    """
    
    # Convert numpy arrays to tensors if needed
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    # Ensure inputs are in the correct shape (batch_num, channels, height, width)
    if y_true.ndim == 3:
        y_true = y_true.unsqueeze(0)
    if y_pred.ndim == 3:
        y_pred = y_pred.unsqueeze(0)
    
    # Calculate MS-SSIM
    ms_ssim_val = ms_ssim(y_true, y_pred, **kwargs)
    
    return 1 - ms_ssim_val

# Example images for testing
# Creating two random images of size (1, 3, 200, 200)
y_true_np = np.random.rand(1, 3, 200, 200).astype(np.float32)
y_pred_np = np.random.rand(1, 3, 200, 200).astype(np.float32)

# Testing ms_ssim_loss with numpy arrays
print("Testing ms_ssim_loss with numpy arrays:")
print(ms_ssim_loss(y_true_np, y_pred_np))

# Creating two random images of size (1, 3, 200, 200) using PyTorch tensors
y_true_torch = torch.rand(1, 3, 200, 200, dtype=torch.float32)
y_pred_torch = torch.rand(1, 3, 200, 200, dtype=torch.float32)

# Testing ms_ssim_loss with PyTorch tensors
print("\nTesting ms_ssim_loss with PyTorch tensors:")
print(ms_ssim_loss(y_true_torch, y_pred_torch))


import torch
import numpy as np

def iou_box_coef(y_true, y_pred, mode='giou', dtype=torch.float32):
    """
    Intersection over Union (IoU) and generalized IoU coefficients for bounding boxes.
    
    Arguments:
    ----------
    y_true: the target bounding box.
    y_pred: the predicted bounding box.
    mode: 'iou' for IoU coefficient (i.e., Jaccard index);
          'giou' for generalized IoU coefficient.
    dtype: the data type of input tensors.
           Default is torch.float32.
    
    Elements of a bounding box should be organized as: [y_min, x_min, y_max, x_max].
    """
    
    zero = torch.tensor(0.0, dtype=dtype)
    
    # unpack bounding box coordinates
    ymin_true, xmin_true, ymax_true, xmax_true = torch.unbind(y_true, dim=-1)
    ymin_pred, xmin_pred, ymax_pred, xmax_pred = torch.unbind(y_pred, dim=-1)
    
    # true area
    w_true = torch.maximum(zero, xmax_true - xmin_true)
    h_true = torch.maximum(zero, ymax_true - ymin_true)
    area_true = w_true * h_true
    
    # pred area
    w_pred = torch.maximum(zero, xmax_pred - xmin_pred)
    h_pred = torch.maximum(zero, ymax_pred - ymin_pred)
    area_pred = w_pred * h_pred
    
    # intersections
    intersect_ymin = torch.maximum(ymin_true, ymin_pred)
    intersect_xmin = torch.maximum(xmin_true, xmin_pred)
    intersect_ymax = torch.minimum(ymax_true, ymax_pred)
    intersect_xmax = torch.minimum(xmax_true, xmax_pred)
    
    w_intersect = torch.maximum(zero, intersect_xmax - intersect_xmin)
    h_intersect = torch.maximum(zero, intersect_ymax - intersect_ymin)
    area_intersect = w_intersect * h_intersect
    
    # IoU
    area_union = area_true + area_pred - area_intersect
    iou = area_intersect / area_union
    
    if mode == "iou":
        return iou
    
    else:
        # enclosed coordinates
        enclose_ymin = torch.minimum(ymin_true, ymin_pred)
        enclose_xmin = torch.minimum(xmin_true, xmin_pred)
        enclose_ymax = torch.maximum(ymax_true, ymax_pred)
        enclose_xmax = torch.maximum(xmax_true, xmax_pred)
        
        # enclosed area
        w_enclose = torch.maximum(zero, enclose_xmax - enclose_xmin)
        h_enclose = torch.maximum(zero, enclose_ymax - enclose_ymin)
        area_enclose = w_enclose * h_enclose
        
        # generalized IoU
        giou = iou - ((area_enclose - area_union) / area_enclose)

        return giou

def iou_box(y_true, y_pred, mode='giou', dtype=torch.float32):
    """
    Intersection over Union (IoU) and generalized IoU losses for bounding boxes.
    
    Arguments:
    ----------
    y_true: the target bounding box.
    y_pred: the predicted bounding box.
    mode: 'iou' for IoU coefficient (i.e., Jaccard index);
          'giou' for generalized IoU coefficient.
    dtype: the data type of input tensors.
           Default is torch.float32.
    
    Elements of a bounding box should be organized as: [y_min, x_min, y_max, x_max].
    """
    
    y_pred = torch.tensor(y_pred, dtype=dtype)
    y_true = torch.tensor(y_true, dtype=dtype)
    
    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()

    return 1 - iou_box_coef(y_true, y_pred, mode=mode, dtype=dtype)

# Example usage

# Test case 1: Overlapping bounding boxes
y_true_1 = np.array([[0.1, 0.1, 0.4, 0.4]])
y_pred_1 = np.array([[0.2, 0.2, 0.5, 0.5]])

# Test case 2: No overlap
y_true_2 = np.array([[0.1, 0.1, 0.2, 0.2]])
y_pred_2 = np.array([[0.3, 0.3, 0.4, 0.4]])

# Test case 3: Perfect match
y_true_3 = np.array([[0.199, 0.2, 0.4, 0.4]])
y_pred_3 = np.array([[0.2, 0.2, 0.4, 0.4]])

print("Test case 1: Overlapping bounding boxes")
print("IoU:", iou_box(y_true_1, y_pred_1, mode='iou').item())
print("GIoU:", iou_box(y_true_1, y_pred_1, mode='giou').item())

print("\nTest case 2: No overlap")
print("IoU:", iou_box(y_true_2, y_pred_2, mode='iou').item())
print("GIoU:", iou_box(y_true_2, y_pred_2, mode='giou').item())

print("\nTest case 3: Perfect match")
print("IoU:", iou_box(y_true_3, y_pred_3, mode='iou').item())
print("GIoU:", iou_box(y_true_3, y_pred_3, mode='giou').item())


def triplet_1d(y_true, y_pred, N, margin=5.0):
    """
    Semi-hard triplet loss with one-dimensional vectors of anchor, positive, and negative.
    
    Arguments:
    ----------
    y_true: a dummy input, not used within this function. Appeared as a requirement of tf.keras.loss function format.
    y_pred: a single pass of triplet training, with shape=(batch_num, 3*embeded_vector_size).
            i.e., y_pred is the ordered and concatenated anchor, positive, and negative embeddings.
    N: Size (dimensions) of embedded vectors.
    margin: a positive number that prevents negative loss.
    """
    
    # anchor sample pair separations.
    Embd_anchor = y_pred[:, 0:N]
    Embd_pos = y_pred[:, N:2*N]
    Embd_neg = y_pred[:, 2*N:]
    
    # squared distance measures
    d_pos = torch.sum((Embd_anchor - Embd_pos) ** 2, dim=1)
    d_neg = torch.sum((Embd_anchor - Embd_neg) ** 2, dim=1)
    loss_val = torch.maximum(torch.tensor(0.0), margin + d_pos - d_neg)
    loss_val = torch.mean(loss_val)
    
    return loss_val

# Example usage
y_pred_triplet = torch.rand((10, 3 * 128))  # 10 samples with 128-dimensional embeddings for anchor, pos, neg
N = 128
print(triplet_1d(None, y_pred_triplet, N, margin=5.0))





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