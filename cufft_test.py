#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 23:07:45 2017

@author: hkohr
"""
import numpy as np
import pyfftw

# Import skcuda FFT module (import is lazy)
import skcuda
import skcuda.fft

# Import and initialize pycuda (context etc.)
import pycuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

# Import pygpu and create context
import pygpu
ctx = pygpu.init('cuda')

# Test 1: Create pygpu.gpuarray.GpuArray, hand over pointer to pycuda
# and run skcuda FFT
# Limitation: pygpu doesn't support complex yet, so no way to get a valid
# array back from pycuda
x_npy = np.arange(24, dtype=float).reshape((2, 3, 4))
x_pygpu = pygpu.gpuarray.array(x_npy, context=ctx)
x_pycuda = gpuarray.GPUArray(shape=x_pygpu.shape, dtype=x_pygpu.dtype,
                             gpudata=x_pygpu.gpudata, strides=x_pygpu.strides)

out = gpuarray.empty(shape=(2, 3, 3), dtype=np.dtype('complex128'))
plan = skcuda.fft.Plan(x_pycuda.shape, x_pycuda.dtype, np.dtype('complex128'))
skcuda.fft.fft(x_pycuda, out, plan)
print(out)
out_npy = out.get()
print(out_npy)

# Test 2: speed of R2C CPU FFT against CUDA FFT
N = 384
shape = (N, N, N)
out_shape = (N, N, N // 2 + 1)
dtype_r = np.dtype('float32')
dtype_c = np.result_type(dtype_r, 1j)

x_npy = np.zeros(shape, dtype_r)
out_npy = np.empty(out_shape, dtype=dtype_c)
x_gpu = gpuarray.to_gpu(x_npy)
out_gpu = gpuarray.empty(out_shape, dtype=dtype_c)
gpu_plan = skcuda.fft.Plan(x_npy.shape, x_npy.dtype, out_npy.dtype)
cpu_plan = pyfftw.FFTW(x_npy, out_npy)


def gpu_fft(x_gpu, out_gpu):
    """Init plan, run FFT, no copies."""
    plan = skcuda.fft.Plan(x_gpu.shape, x_gpu.dtype, out_gpu.dtype)
    skcuda.fft.fft(x_gpu, out_gpu, plan)


def gpu_fft_with_result_copy(x_gpu, out_gpu, out_cpu):
    """Init plan, run FFT, copy result to CPU."""
    plan = skcuda.fft.Plan(x_gpu.shape, x_gpu.dtype, out_gpu.dtype)
    skcuda.fft.fft(x_gpu, out_gpu, plan)
    out_gpu.get(out_cpu)


def gpu_fft_with_both_copies(x_cpu, out_gpu, out_cpu):
    """Init plan, run FFT, copying input and result from/to CPU."""
    x_gpu = gpuarray.to_gpu(x_cpu)
    plan = skcuda.fft.Plan(x_gpu.shape, x_gpu.dtype, out_gpu.dtype)
    skcuda.fft.fft(x_gpu, out_gpu, plan)
    out_gpu.get(out_cpu)


def gpu_fft_with_both_copies_and_temp(x_cpu, out_cpu):
    """Init plan, create out, run FFT, copying input and result from/to CPU."""
    x_gpu = gpuarray.to_gpu(x_cpu)
    out_shape = list(x_gpu.shape)
    out_shape[-1] = x_gpu.shape[-1] // 2 + 1
    out_gpu = gpuarray.empty(out_shape, dtype=out_cpu.dtype)
    plan = skcuda.fft.Plan(x_gpu.shape, x_gpu.dtype, out_gpu.dtype)
    skcuda.fft.fft(x_gpu, out_gpu, plan)
    out_gpu.get(out_cpu)


# Run these in IPython
# %timeit skcuda.fft.fft(x_gpu, out_gpu, gpu_plan)  # pure fft call
# %timeit gpu_fft(x_gpu, out_gpu)  # with plan creation
# %timeit gpu_fft_with_result_copy(x_gpu, out_gpu, out_npy)
# %timeit gpu_fft_with_both_copies(x_npy, out_gpu, out_npy)
# %timeit gpu_fft_with_both_copies_and_temp(x_npy, out_npy)
# %timeit cpu_plan()
