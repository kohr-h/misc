#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:10:34 2017

@author: kohr
"""

from chainer import Chain, Variable
from chainer.links import ConvolutionND
from chainer.functions import relu, mean_squared_error
import cupy


class TestChain3d(Chain):

    def __init__(self):
        super(TestChain3d, self).__init__()
        with self.init_scope():
            self.conv1 = ConvolutionND(3, in_channels=1, out_channels=4,
                                       ksize=[3, 3, 3], nobias=True)
            self.conv2 = ConvolutionND(3, in_channels=4, out_channels=8,
                                       ksize=[3, 3, 3], nobias=True)
            self.conv3 = ConvolutionND(3, in_channels=8, out_channels=1,
                                       ksize=[3, 3, 3], nobias=True)

    def __call__(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.conv3(x)
        x = relu(x)
        return x


N = 128
cupy.zeros(1)  # Init CUDA to set the baseline memory usage

# %% Network definition
chain = TestChain3d()
chain.to_gpu()

# %% Variables
x_arr = cupy.zeros((N, N, N), dtype='float32')
print('Input size (MB): {}'.format(x_arr.size * x_arr.itemsize / 1e6))
x = Variable(x_arr)[None, None, ...]  # empty batch and channel axes
# Convolution doesn't pad, so size shrinks by `kernel.shape - 1` each time
tgt_arr = cupy.zeros((N - 6, N - 6, N - 6), dtype='float32')
tgt = Variable(tgt_arr)[None, None, ...]  # empty batch and channel axes

# %% Forward
y = chain(x)
z = mean_squared_error(y, tgt)

# %% Backward
z.backward()
