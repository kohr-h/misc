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

ufunc = 'logaddexp2'
dtype1 = 'float32'
dtype2 = 'float32'


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


x_npy = random_array(dtype1)
y_npy = random_array(dtype2)
x_pygpu = pygpu.array(x_npy, dtype=dtype1)
y_pygpu = pygpu.array(y_npy, dtype=dtype2)
ufunc_npy = getattr(np, ufunc)
ufunc_pygpu = getattr(pygpu.ufuncs, ufunc)
res_npy = ufunc_npy(x_npy, y_npy)
res_pygpu = ufunc_pygpu(x_pygpu, y_pygpu)

print('=== testing ufunc {} for dtypes {}, {} ==='
      ''.format(ufunc, dtype1, dtype2))
print('x =', x_npy)
print('y =', y_npy)
print('npy:   {}(x, y) ='.format(ufunc))
print(res_npy)
print('pygpu: {}(x, y) ='.format(ufunc))
print(res_pygpu)
