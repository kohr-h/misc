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
from time import time


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


N = 512

chain = TestChain3d()
chain.to_gpu()

x_arr = cupy.zeros((N, N, N), dtype='float32')
x = Variable(x_arr)[None, None, ...]  # empty batch and channel axes
tgt_arr = cupy.zeros((N - 6, N - 6, N - 6), dtype='float32')
tgt = Variable(tgt_arr)[None, None, ...]  # empty batch and channel axes

t_start = time()
y = chain(x)
t_after_chain = time()
z = mean_squared_error(y, tgt)
t_after_loss = time()
z.backward()
t_after_grad = time()

print('Time for forward pass: {:.4} s'
      ''.format(t_after_chain - t_start))
print('Time for loss computation: {:.4} s'
      ''.format(t_after_loss - t_after_chain))
print('Time for gradient computation: {:.4} s'
      ''.format(t_after_grad - t_after_loss))
