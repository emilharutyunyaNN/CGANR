import numpy as np
import torch
import torch.nn as nn
from aligners.utils import transform, resize, integrate_vec, affine_to_shift


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
            self._initialize(x)

        vol, trf = x
        if self.is_affine:
            #print("affine")
            trf = torch.stack([self._single_aff_to_shift(t, vol.shape[2:]) for t in trf])
        
        if self.indexing == 'xy':
            #print("xy")
            trf = torch.cat((trf[..., 1:2], trf[..., 0:1], trf[..., 2:]), dim=-1)

        if self.single_transform:
           # print("single transform")
            transformed = [self._single_transform(vol[i], trf[0]) for i in range(vol.size(0))]
            return torch.stack(transformed)
        else:
           # print("else")
           # print(vol.size(0))
            return torch.stack([self._single_transform(vol[i], trf[i]) for i in range(vol.size(0))])

    def _initialize(self, inputs):
        input_shape = [input.size() for input in inputs]
        if len(input_shape) != 2:
            raise Exception('Spatial Transformer must be called on a list of length 2.')

        self.ndims = len(input_shape[0]) - 2
        #print("dims: ", self.ndims)
        #print("input shape 1: ", input_shape[1])
        vol_shape = input_shape[0][2:]
        trf_shape = input_shape[1][1:]

        self.is_affine = len(trf_shape) == 1 or (len(trf_shape) == 2 and all(f == (self.ndims + 1) for f in trf_shape))

        if self.is_affine and len(trf_shape) == 1:
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
        trf += torch.eye(self.ndims + 1, device=trf.device)[:self.ndims]
        return affine_to_shift(trf, volshape, shift_center=True)

    def _single_transform(self, vol, trf):
        print(vol.shape, trf.shape)
        vol = vol.permute(1,2,0)
        trf = trf.permute(1,2,0)
       # print("vol trf: ----", vol.shape, trf.shape, vol.device, trf.device)
        return transform(vol, trf, interp_method=self.interp_method)