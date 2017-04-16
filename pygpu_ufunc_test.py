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

ufunc = 'floor_divide'
dtype1 = 'uint64'
dtype2 = 'bool'

x_npy = np.random.randint(low=-2, high=3, size=10).astype(dtype1)
y_npy = np.random.randint(low=-2, high=3, size=10).astype(dtype2)
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
print('npy:   {}(x) ='.format(ufunc))
print(res_npy)
print('pygpu: {}(x) ='.format(ufunc))
print(res_pygpu)
