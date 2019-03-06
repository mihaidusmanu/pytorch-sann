from itertools import product

import numpy as np

import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sann import spatially_aware_nn

import time

grad_dtypes = [torch.float]

if torch.cuda.is_available():
    devices = [torch.device('cuda:{}'.format(torch.cuda.current_device()))]
else:
    raise NotImplemented('Please use CUDA for spatially aware nearest neighbors.')

def tensor(x, dtype, device):
    return None if x is None else torch.tensor(x, dtype=dtype, device=device)

@pytest.mark.parametrize('dtype,device', product(grad_dtypes, devices))
def test_spatially_aware_nn(dtype, device):
    N = 16384
    Fdim = 64
    x = tensor(np.random.randn(N, Fdim), dtype, device)
    x = F.normalize(x, dim=1)

    y = tensor(np.random.randn(N, Fdim), dtype, device)
    y = F.normalize(y, dim=1)
    
    pos_x = tensor(np.random.randn(N, 2), dtype, device)

    pos_y = tensor(np.random.randn(N, 2), dtype, device)

    pos_dist_threshold = 0.5

    t0 = time.time()
    nn_idx = spatially_aware_nn(x, y, pos_x, pos_y, pos_dist_threshold)
    print('GPU runtime %.4f;' % (time.time() - t0), end=' ')

    cpu_x = x.data.cpu().numpy()
    
    cpu_y = y.data.cpu().numpy()
    
    cpu_pos_x = pos_x.data.cpu().numpy()
    
    cpu_pos_y = pos_y.data.cpu().numpy()
    
    t0 = time.time()
    pos_dist = np.max(np.abs(cpu_pos_x[:, np.newaxis, :] - cpu_pos_y[np.newaxis, :, :]), axis=2)
    valid = pos_dist > pos_dist_threshold
    pos_dist = None
    descr_sim = np.dot(cpu_x, np.transpose(cpu_y))
    minf = -10
    cpu_nn_idx = np.argmax(minf * (1 - valid) + descr_sim, axis=1)
    print('CPU runtime %.4f' % (time.time() - t0), end=' ')

    assert np.array_equal(nn_idx.data.cpu().numpy(), cpu_nn_idx)
