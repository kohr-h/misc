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

ufunc = 'hypot'
dtype1 = 'uint16'
dtype2 = 'uint16'
scalar = -2
scalar_pos = 'left'


def random_array(dtype):
    dtype = np.dtype(dtype)
    if dtype == bool:
        return np.random.randint(low=0, high=2, size=10).astype(bool)
    elif np.issubsctype(dtype, np.unsignedinteger):
        return np.random.randint(low=0, high=5, size=10).astype(dtype)
    elif np.issubsctype(dtype, np.signedinteger):
        return np.random.randint(low=-4, high=5, size=10).astype(dtype)
    elif np.issubsctype(dtype, np.floating):
        return np.random.uniform(low=-4, high=4, size=10).astype(dtype)
    else:
        raise ValueError('unable to handle dtype {}'.format(dtype))


arr_dtype = dtype1 if scalar_pos == 'right' else dtype2
x_npy = random_array(arr_dtype)
x_pygpu = pygpu.array(x_npy, dtype=arr_dtype)

ufunc_npy = getattr(np, ufunc)
ufunc_pygpu = getattr(pygpu.ufuncs, ufunc)
if scalar_pos == 'left':
    res_npy = ufunc_npy(scalar, x_npy)
    res_pygpu = ufunc_pygpu(scalar, x_pygpu)
else:
    res_npy = ufunc_npy(x_npy, scalar)
    res_pygpu = ufunc_pygpu(x_pygpu, scalar)

print('=== testing ufunc {} for dtype {} ==='.format(ufunc, arr_dtype))
print('x =', x_npy)
print('npy:   {}(...) = ({})'.format(ufunc, res_npy.dtype))
print(res_npy)
print('pygpu: {}(...) = ({})'.format(ufunc, res_pygpu.dtype))
print(res_pygpu)
