#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:10:34 2017

@author: kohr
"""

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Module, Conv3d, ReLU
from torch.nn.functional import mse_loss


class TestModule3d(Module):

    def __init__(self):
        super(TestModule3d, self).__init__()
        self.conv1 = Conv3d(in_channels=1, out_channels=4,
                            kernel_size=[3, 3, 3], bias=False)
        self.nonlin1 = ReLU()
        self.conv2 = Conv3d(in_channels=4, out_channels=8,
                            kernel_size=[3, 3, 3], bias=False)
        self.nonlin2 = ReLU()
        self.conv3 = Conv3d(in_channels=8, out_channels=1,
                            kernel_size=[3, 3, 3], bias=False)
        self.nonlin3 = ReLU()

    def __call__(self, x):
        x = self.conv1(x)
        x = self.nonlin1(x)
        x = self.conv2(x)
        x = self.nonlin2(x)
        x = self.conv3(x)
        x = self.nonlin3(x)
        return x


N = 64
torch.cuda.FloatTensor(1)  # Init CUDA to set the baseline memory usage

# %% Network definition
net = TestModule3d()
net.cuda()

# %% Variables
x_arr = torch.zeros((N, N, N))
print('Input size (MB): {}'
      ''.format(np.prod(x_arr.size()) * x_arr.element_size() / 1e6))
x = Variable(x_arr)[None, None, ...].cuda()  # empty batch and channel axes
# Convolution doesn't pad, so size shrinks by `kernel.shape - 1` each time
tgt_arr = torch.zeros((N - 6, N - 6, N - 6))
tgt = Variable(tgt_arr)[None, None, ...].cuda()  # empty batch and channel axes

# %% Forward
y = net(x)
z = mse_loss(y, tgt)

# %% Backward
z.backward()
