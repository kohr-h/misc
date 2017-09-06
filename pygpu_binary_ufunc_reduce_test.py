#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 00:20:55 2017

@author: hkohr
"""
import numpy as np
import pygpu
ctx = pygpu.init('cuda')
pygpu.set_default_context(ctx)

ufunc = 'logical_and'
axis = 0
keepdims = True
dtype = 'float32'


def random_array(dtype):
    dtype = np.dtype(dtype)
    if dtype == bool:
        return np.random.randint(low=0, high=2, size=10).astype(bool)
    elif np.issubsctype(dtype, np.unsignedinteger):
        return np.random.randint(low=0, high=10, size=10).astype(dtype)
    elif np.issubsctype(dtype, np.signedinteger):
        return np.random.randint(low=0, high=10, size=10).astype(dtype)
    elif np.issubsctype(dtype, np.floating):
        return np.random.uniform(low=-4, high=4, size=10).astype(dtype)
    else:
        raise ValueError('unable to handle dtype {}'.format(dtype))


x_npy = random_array(dtype)
x_pygpu = pygpu.array(x_npy, dtype=dtype)
ufunc_npy = getattr(np, ufunc)
ufunc_pygpu = getattr(pygpu.ufuncs, ufunc)
res_npy = ufunc_npy.reduce(x_npy, axis=axis, keepdims=keepdims)
res_pygpu = ufunc_pygpu.reduce(x_pygpu, axis=axis, keepdims=keepdims)

print('=== testing reduce of ufunc {} for dtype {} ==='
      ''.format(ufunc, dtype))
print('x =', x_npy)
print('npy:   {}.reduce(x) ='.format(ufunc))
print(res_npy)
print('pygpu: {}.reduce(x) ='.format(ufunc))
print(res_pygpu)
