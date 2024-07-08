import itertools
import torch
import torchvision.transforms as transforms
# third party imports
import numpy as np
from tqdm import tqdm_notebook as tqdm
from pprint import pformat

import torch
import itertools

def interpn(vol, loc, interp_method='linear'):
    if isinstance(loc, (list, tuple)):
        loc = torch.stack(loc, dim=-1)
    #print("loc: ", loc.shape)
    nb_dims = loc.shape[-1]
    
    if nb_dims != len(vol.shape[:-1]):
        raise ValueError("Number of loc Tensors {} does not match volume dimension {}".format(nb_dims, len(vol.shape[:-1])))
    
    if len(vol.shape) == nb_dims:
        vol = vol.unsqueeze(-1)
    
    loc = loc.float()

    if interp_method == 'linear':
        loc0 = torch.floor(loc)
        max_loc = [d - 1 for d in vol.shape[:-1]]
        clipped_loc = [torch.clamp(loc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [torch.clamp(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        loc1 = [torch.clamp(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[f.int() for f in loc0lst], [f.int() for f in loc1]]
        
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        weights_loc = [diff_loc1, diff_loc0]
        
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0
        
        for c in cube_pts:
            subs = [locs[c[d]][d] for d in range(nb_dims)]
            idx = sub2ind(vol.shape[:-1], subs)
            vol_val = vol.view(-1, vol.shape[-1])[idx]
            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            wt = prod_n(wts_lst)
            wt = wt.unsqueeze(-1)
            interp_vol += wt * vol_val
    
    elif interp_method == 'nearest':
        roundloc = loc.round().int()
        max_loc = [d - 1 for d in vol.shape[:-1]]
        roundloc = [torch.clamp(roundloc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        idx = sub2ind(vol.shape[:-1], roundloc)
        interp_vol = vol.view(-1, vol.shape[-1])[idx]
    
    return interp_vol

def sub2ind(siz, subs):
    k = torch.cumprod(torch.tensor(siz[::-1]), 0)
    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]
    return ndx

def prod_n(lst):
    prod = lst[0]
    for p in lst[1:]:
        prod *= p
    return prod


def resize(vol, zoom_factor, interp_method='linear'):
    if isinstance(zoom_factor, (list, tuple)):
        ndims = len(zoom_factor)
        vol_shape = vol.shape[:ndims]
        assert len(vol_shape) in (ndims, ndims+1), "zoom_factor length {} does not match ndims {}".format(len(vol_shape), ndims)
    else:
        vol_shape = vol.shape[:-1]
        ndims = len(vol_shape)
        zoom_factor = [zoom_factor] * ndims
    
    if not isinstance(vol_shape[0], int):
        vol_shape = vol_shape.tolist()
    
    new_shape = [int(vol_shape[f] * zoom_factor[f]) for f in range(ndims)]
    grid = volshape_to_ndgrid(new_shape)
    grid = [f.float() for f in grid]
    offset = [grid[f] / zoom_factor[f] - grid[f] for f in range(ndims)]
    offset = torch.stack(offset, dim=ndims)
    return transform(vol, offset, interp_method)

def transform(vol, loc_shift, interp_method='linear', indexing='ij'):
    if isinstance(loc_shift.shape, (torch.Size, tuple)):
        volshape = loc_shift.shape[:-1]
    else:
        volshape = loc_shift.shape[:-1]
    #print("vol: ", vol.shape, volshape)
    nb_dims = len(volshape)
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)
    mesh = [m.to(loc_shift.device) for m in mesh]
    #print("mesh: ", len(mesh))
    loc = [mesh[d].float() + loc_shift[..., d] for d in range(nb_dims)]
    return interpn(vol, loc, interp_method=interp_method)

def volshape_to_ndgrid(volshape, **kwargs):
    linvec = [torch.arange(0, d) for d in volshape]
    return ndgrid(*linvec, **kwargs)

def volshape_to_meshgrid(volshape, **kwargs):
    linvec = [torch.arange(0, d) for d in volshape]
    return meshgrid(*linvec, **kwargs)

def ndgrid(*args, **kwargs):
    return meshgrid(*args, indexing='ij', **kwargs)

def meshgrid(*args, **kwargs):
    indexing = kwargs.pop("indexing", "xy")
    ndim = len(args)
    s0 = (1,) * ndim
    output = []
    for i, x in enumerate(args):
        output.append(x.view(s0[:i] + (-1,) + s0[i + 1:]))
    shapes = [torch.numel(x) for x in args]
    sz = [x.size(0) for x in args]
    
    if indexing == "xy" and ndim > 1:
        output[0] = output[0].view((1, -1) + (1,) * (ndim - 2))
        output[1] = output[1].view((-1, 1) + (1,) * (ndim - 2))
        shapes[0], shapes[1] = shapes[1], shapes[0]
        sz[0], sz[1] = sz[1], sz[0]
    
    for i in range(len(output)):
        output[i] = output[i].repeat(*sz[:i], 1, *sz[(i+1):])
    return output

from torchdiffeq import odeint

def integrate_vec(vec, time_dep=False, method='ss', **kwargs):
    if method not in ['ss', 'scaling_and_squaring', 'ode', 'quadrature']:
        raise ValueError("method has to be 'scaling_and_squaring' or 'ode'. found: %s" % method)

    if method in ['ss', 'scaling_and_squaring']:
        nb_steps = kwargs['nb_steps']
        assert nb_steps >= 0, 'nb_steps should be >= 0, found: %d' % nb_steps

        if time_dep:
            svec = vec.permute(-1, *range(vec.ndimension() - 1))
            assert 2 ** nb_steps == svec.shape[0], "2**nb_steps and vector shape don't match"

            svec = svec / (2 ** nb_steps)
            for _ in range(nb_steps):
                svec = svec[0::2] + torch.stack([transform(svec[i + 1], svec[i]) for i in range(0, svec.shape[0] - 1, 2)])

            disp = svec[0]

        else:
            vec = vec / (2 ** nb_steps)
            for _ in range(nb_steps):
                vec = vec + transform(vec, vec)
            disp = vec

    elif method == 'quadrature':
        nb_steps = kwargs['nb_steps']
        assert nb_steps >= 1, 'nb_steps should be >= 1, found: %d' % nb_steps

        vec = vec / nb_steps

        if time_dep:
            disp = vec[..., 0]
            for si in range(nb_steps - 1):
                disp += transform(vec[..., si + 1], disp)
        else:
            disp = vec
            for _ in range(nb_steps - 1):
                disp += transform(vec, disp)

    elif method == 'ode':
        assert not time_dep, "odeint not implemented with time-dependent vector field"
        fn = lambda t, disp: transform(vec, disp)

        out_time_pt = kwargs.get('out_time_pt', 1)
        if not isinstance(out_time_pt, (list, tuple)):
            out_time_pt = [out_time_pt]

        out_time_pt = torch.tensor([0] + out_time_pt, dtype=torch.float32)

        init = kwargs.get('init', 'zero')
        if init == 'zero':
            disp0 = torch.zeros_like(vec)
        else:
            raise ValueError('non-zero init for ode method not implemented')

        ode_args = kwargs.get('ode_args', {})
        disp = odeint(fn, disp0, out_time_pt, **ode_args)
        disp = disp.permute([*range(1, disp.ndimension()), 0])

        if len(out_time_pt) == 2:
            disp = disp[..., 1]

    return disp

def affine_to_shift(affine_matrix, volshape, shift_center=True, indexing='ij'):
    """
    Transform an affine matrix to a dense location shift tensor in PyTorch.

    Algorithm:
        - Get grid and shift grid to be centered at the center of the image (optionally)
        - Apply affine matrix to each index.
        - Subtract grid

    Parameters:
        affine_matrix: ND+1 x ND+1 or ND x ND+1 matrix (Tensor)
        volshape: 1xN Nd Tensor of the size of the volume.
        shift_center (optional)

    Returns:
        Shift field (Tensor) of size *volshape x N

    TODO: 
        Allow affine_matrix to be a vector of size nb_dims * (nb_dims + 1)
    """
    if isinstance(volshape, torch.Size) or isinstance(volshape, tuple):
        volshape = list(volshape)
    else:
        raise ValueError('volshape should be a list or tuple of integers.')
    affine_matrix = affine_matrix.float()
    
    nb_dims = len(volshape)
    
    if len(affine_matrix.shape) == 1:
        if len(affine_matrix) != (nb_dims * (nb_dims + 1)):
            raise ValueError('transform is supposed a vector of len ndims * (ndims + 1).'
                             'Got len %d' % len(affine_matrix))
        affine_matrix = affine_matrix.view(nb_dims, nb_dims + 1)
        
    if not (affine_matrix.shape[0] in [nb_dims, nb_dims + 1] and affine_matrix.shape[1] == (nb_dims + 1)):
        print(affine_matrix.shape)
        raise Exception('Affine matrix shape should match'
                        '%d+1 x %d+1 or ' % (nb_dims, nb_dims) + \
                        '%d x %d+1.' % (nb_dims, nb_dims) + \
                        'Got: ' + str(volshape))
    
    # Create meshgrid for the volume shape
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)
    mesh = [f.float() for f in mesh]
    
    if shift_center:
        mesh = [mesh[d] - (volshape[d] - 1) / 2 for d in range(nb_dims)]
        
    # Add an all-ones entry and transform into a large matrix
    flat_mesh = [flatten(f) for f in mesh]
    flat_mesh.append(torch.ones(flat_mesh[0].shape, dtype=torch.float32))
    mesh_matrix = torch.stack(flat_mesh, dim=1).transpose(0, 1)
    
    # Compute locations
    loc_matrix = torch.matmul(affine_matrix, mesh_matrix)
    loc_matrix = loc_matrix[:nb_dims, :].transpose(0, 1)
    
    loc = loc_matrix.view(*volshape, nb_dims)
    
    return loc - torch.stack(mesh, dim=nb_dims)

def batch_affine_to_shift(affine_matrix, volshape, shift_center=True, indexing='ij', batch_size=16):
    """
    transform an affine matrix to a dense location shift tensor in tensorflow

    Algorithm:
        - get grid and shift grid to be centered at the center of the image (optionally)
        - apply affine matrix to each index.
        - subtract grid

    Parameters:
        affine_matrix: B x ND+1 x ND+1 or B x (ND x ND+1) matrix (Tensor)
        volshape: 1xN Nd Tensor of the size of the volume.
        shift_center (optional)

    Returns:
        shift field (Tensor) of size  B x (*volshape) x N
    """
    if isinstance(volshape, torch.Size) or isinstance(volshape, tuple):
        volshape = list(volshape)
    else:
        raise ValueError('volshape should be a list or tuple of integers.')
    affine_matrix = affine_matrix.float()
    
    nb_dims = len(volshape)
    if len(affine_matrix.shape) == 2:
        affine_matrix = affine_matrix.view(-1, nb_dims, nb_dims + 1)
        
    if not (affine_matrix.shape[1] in [nb_dims, nb_dims + 1] and affine_matrix.shape[2] == (nb_dims + 1)):
        print(affine_matrix.shape)
        raise Exception('Affine matrix shape should match'
                        '%d+1 x %d+1 or ' % (nb_dims, nb_dims) + \
                        '%d x %d+1.' % (nb_dims, nb_dims) + \
                        'Got: ' + str(volshape))
    shifts = []
    for b in range(batch_size):
        # list of volume ndgrid
        # N-long list, each entry of shape volshape
        mesh = volshape_to_meshgrid(volshape, indexing=indexing)
        mesh = [f.flaot() for f in mesh]
        
        if shift_center:
            mesh = [mesh[f] - (volshape[f] - 1) / 2 for f in range(len(volshape))]
             
        flat_mesh = [flatten(f) for f in mesh]
        flat_mesh.append(torch.ones(flat_mesh[0].shape, dtype=torch.float32))
        mesh_matrix = torch.stack(flat_mesh, dim=1).transpose(0, 1)     

        loc_matrix = torch.matmul(affine_matrix, mesh_matrix)
        loc_matrix = loc_matrix[:nb_dims, :].transpose(0, 1)
        loc = loc_matrix.view(*volshape, nb_dims)

        shifts.append(loc - torch.stack(mesh, dim=nb_dims))
        
    return torch.stack(shifts, dim=0)
def flatten(v):
    
    return v.reshape(-1)


