import numpy as np
import torch
import torch.nn as nn
from aligners.utils import transform, resize, integrate_vec, affine_to_shift
import matplotlib.pyplot as plt
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    def __init__(self, interp_method='linear', indexing='ij', single_transform=False):
        super(SpatialTransformer, self).__init__()
        self.interp_method = interp_method
        self.indexing = indexing
        self.single_transform = single_transform
        self.is_affine = None
        self.ndims = None
        self.initialized = False

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"

    def forward(self, x):
        if not self.initialized:
            #print("init")
            self._initialize(x)

        vol, trf = x
        if self.is_affine:
            #print("affine")
            trf = torch.stack([self._single_aff_to_shift(t, vol.shape[1:-1]) for t in trf])
        
        if self.indexing == 'xy':
           # print("xy")
            trf = torch.cat((trf[..., 1:2], trf[..., 0:1], trf[..., 2:]), dim=-1)

        if self.single_transform:
            #print("single transform")
            transformed = [self._single_transform(vol[i], trf[0]) for i in range(vol.size(0))]
            return torch.stack(transformed)
        else:

            #print(trf.shape, vol.shape)
           # print("else")
            #print(vol.size(0))
            #print("***",torch.stack([self._single_transform(vol[i], trf[i]) for i in range(vol.size(0))]).shape)
            return torch.stack([self._single_transform(vol[i], trf[i]) for i in range(vol.size(0))]), trf

    def _initialize(self, inputs):
        input_shape = [input.size() for input in inputs]
        if len(input_shape) != 2:
            raise Exception('Spatial Transformer must be called on a list of length 2.')

        self.ndims = len(input_shape[0]) - 2
        #print("dims: ", self.ndims)
        #print("input shape 1: ", input_shape[1])
        vol_shape = input_shape[0][2:]
        trf_shape = input_shape[1][1:]
        # print("trrf: ", trf_shape)
        self.is_affine = len(trf_shape) == 1 or (len(trf_shape) == 2 and all(f == (self.ndims + 1) for f in trf_shape))

        if self.is_affine and len(trf_shape) == 1:
            print("affine")
            ex = self.ndims * (self.ndims + 1)
            if trf_shape[0] != ex:
                raise Exception(f'Expected flattened affine of length {ex} but got {trf_shape[0]}')

        if not self.is_affine:
            #print("trf:", trf_shape)
            if trf_shape[0] != self.ndims:
                raise Exception(f'Offset flow field size expected: {self.ndims}, found: {trf_shape[-1]}')

        self.initialized = True

    def _single_aff_to_shift(self, trf, volshape):
        if trf.ndim == 1:
            trf = trf.view(self.ndims, self.ndims + 1)
        trf = trf + torch.eye(self.ndims + 1, device=trf.device)[:self.ndims]
        return affine_to_shift(trf, volshape, shift_center=True)

    """def _single_transform(self, vol, trf):
        #print(vol.shape, trf.shape)
        vol = vol.permute(1,2,0)
        trf = trf.permute(1,2,0)
       # print("vol trf: ----", vol.shape, trf.shape, vol.device, trf.device)
        #vol_old = np.clip(vol.detach().clone().cpu().numpy().astype(np.float32),0,1)
       # plt.imsave("./voxelmorph_reg/vol_old.jpg", vol_old)
        vol_new =  transform(vol, trf, interp_method=self.interp_method)
        print(type(vol_new), vol_new.shape)
        vol_new_1 = np.clip(vol_new.detach().clone().cpu().numpy().astype(np.float32),0,1)
        print(vol_new_1)
        plt.imshow(vol_new_1)
        plt.imsave("./voxelmorph_reg/vol_new.jpg", vol_new_1)
        plt.close()
        return vol_new"""
        
    def _single_transform(self, vol, trf):
        #self.affine = True
        if vol.shape[-1] == 1 or vol.shape[-1] == 3:
            vol = vol.permute(2,0,1).contiguous()
            trf = trf.permute(2,0,1).contiguous()
        #print("---///", vol.shape, trf.shape)
        
        trf = trf.permute(1, 2, 0).unsqueeze(0)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, vol.shape[1] - 1, vol.shape[1]),
            torch.linspace(0, vol.shape[2] - 1, vol.shape[2]),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).to(vol.device)

        # Add the transformation field to the mesh grid
        #print(grid.shape)
        sampling_grid = grid + trf

        # Convert the resulting grid to the range [-1, 1]
        sampling_grid[..., 0] = 2.0 * (sampling_grid[..., 0] / (vol.shape[2] - 1)) - 1.0
        sampling_grid[..., 1] = 2.0 * (sampling_grid[..., 1] / (vol.shape[1] - 1)) - 1.0

        vol = vol.unsqueeze(0)
        vol_new = F.grid_sample(vol, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)

        # Save the old volume before transformation
        vol_old = vol.squeeze(0).permute(1, 2, 0).detach().clone().cpu().numpy().astype(np.float32)
        vol_old_clipped = np.clip(vol_old, 0, 1)
       # print(vol_old.shape)
        if self.is_affine:
            vol_old_clipped = vol_old_clipped.squeeze()
            plt.imsave(f"./voxelmorph_reg/vol_old.jpg", vol_old_clipped, cmap = 'gray')

        # Save the new volume after transformation
        vol_new = vol_new.squeeze(0)
        vol_new_clipped = np.clip(vol_new.permute(1, 2, 0).detach().clone().cpu().numpy().astype(np.float32), 0, 1)
        if self.is_affine:
            vol_new_clipped = vol_new_clipped.squeeze()
            plt.imsave(f"./voxelmorph_reg/vol_new.jpg", vol_new_clipped, cmap = 'gray')

        # Plot overlapping images for comparison
        """plt.close()
        plt.imshow(vol_old_clipped, cmap='gray', alpha=0.5)
        plt.imshow(vol_new_clipped, cmap='hot', alpha=0.5)
        plt.savefig("./voxelmorph_reg/overlap_R.jpg")
        plt.close()"""

        return vol_new