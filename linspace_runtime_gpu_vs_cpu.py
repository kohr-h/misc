import numpy as np
import odl
import pygpu
from scipy.linalg import get_blas_funcs
from time import time


dtype = 'float32'
d = 3
ns = [0, 1, 2]
shapes = [(5 * 10 ** n,) * d for n in ns]

scal, axpy = get_blas_funcs(['scal', 'axpy'], dtype=dtype)

a = 2.0
b = -1.5


def print_times_scal():
    print('')
    print('SCAL')
    print('====')
    print('')
    for shape in shapes:
        print('shape = {}'.format(shape))
        x_gpu = pygpu.zeros(shape, dtype=dtype)
        out_gpu = x_gpu._empty_like_me()
        tstart = time()
        odl.space.gpuary_tensors.scal(a, x_gpu, out_gpu)
        tstop = time()
        print('GPU time:            {:.5}'.format(tstop - tstart))

        x_cpu = np.zeros(shape, dtype=dtype)
        out_cpu = np.empty_like(x_cpu)
        tstart = time()
        np.multiply(a, x_cpu, out=out_cpu)
        tstop = time()
        print('CPU time, no copy:   {:.5}'.format(tstop - tstart))

        tstart = time()
        out_cpu[:] = scal(a, x_cpu)
        tstop = time()
        print('BLAS time:           {:.5}'.format(tstop - tstart))

        tstart = time()
        x_gpu_to_cpu = np.asarray(x_gpu)
        np.multiply(a, x_gpu_to_cpu, out=out_cpu)
        out_gpu[:] = out_cpu
        tstop = time()
        print('CPU time, with copy: {:.5}'.format(tstop - tstart))

        print('')


def print_times_axpy():
    print('')
    print('AXPY')
    print('====')
    print('')
    for shape in shapes:
        print('shape = {}'.format(shape))
        x_gpu = pygpu.zeros(shape, dtype=dtype)
        y_gpu = pygpu.zeros(shape, dtype=dtype)
        tstart = time()
        odl.space.gpuary_tensors.axpy(a, x_gpu, y_gpu)
        tstop = time()
        print('GPU time:            {:.5}'.format(tstop - tstart))

        x_cpu = np.zeros(shape, dtype=dtype)
        y_cpu = np.zeros_like(x_cpu)
        out_cpu = np.empty_like(x_cpu)
        tstart = time()
        np.multiply(a, x_cpu, out=out_cpu)
        out_cpu += y_cpu
        tstop = time()
        print('CPU time, no copy:   {:.5}'.format(tstop - tstart))

        tstart = time()
        out_cpu[:] = axpy(x_cpu, y_cpu, a=a)
        tstop = time()
        print('BLAS time:           {:.5}'.format(tstop - tstart))

        out_gpu = x_gpu._empty_like_me()
        tstart = time()
        x_gpu_to_cpu = np.asarray(x_gpu)
        y_gpu_to_cpu = np.asarray(y_gpu)
        np.multiply(a, x_gpu_to_cpu, out=out_cpu)
        out_cpu += y_gpu_to_cpu
        out_gpu[:] = out_cpu
        tstop = time()
        print('CPU time, with copy: {:.5}'.format(tstop - tstart))

        print('')


def print_times_axpby():
    print('')
    print('AXPBY')
    print('=====')
    print('')
    for shape in shapes:
        print('shape = {}'.format(shape))
        x_gpu = pygpu.zeros(shape, dtype=dtype)
        y_gpu = pygpu.zeros(shape, dtype=dtype)
        tstart = time()
        odl.space.gpuary_tensors.axpby(a, x_gpu, b, y_gpu)
        tstop = time()
        print('GPU time:            {:.5}'.format(tstop - tstart))

        x_cpu = np.zeros(shape, dtype=dtype)
        y_cpu = np.zeros_like(x_cpu)
        tstart = time()
        y_cpu *= b
        y_cpu += a * x_cpu
        tstop = time()
        print('CPU time, no copy:   {:.5}'.format(tstop - tstart))

        tstart = time()
        x_gpu_to_cpu = np.asarray(x_gpu)
        y_gpu_to_cpu = b * np.asarray(y_gpu)
        y_gpu_to_cpu += a * x_gpu_to_cpu
        y_gpu[:] = y_gpu_to_cpu
        tstop = time()
        print('CPU time, with copy: {:.5}'.format(tstop - tstart))

        print('')


def print_times_lico():
    print('')
    print('LICO')
    print('====')
    print('')
    for shape in shapes:
        print('shape = {}'.format(shape))
        x_gpu = pygpu.zeros(shape, dtype=dtype)
        y_gpu = pygpu.zeros(shape, dtype=dtype)
        out_gpu = x_gpu._empty_like_me()
        tstart = time()
        odl.space.gpuary_tensors.lico(a, x_gpu, b, y_gpu, out_gpu)
        tstop = time()
        print('GPU time:            {:.5}'.format(tstop - tstart))

        x_cpu = np.zeros(shape, dtype=dtype)
        y_cpu = np.zeros_like(x_cpu)
        out_cpu = np.empty_like(x_cpu)
        tstart = time()
        out_cpu[:] = a * x_cpu
        out_cpu += b * y_cpu
        tstop = time()
        print('CPU time, no copy:   {:.5}'.format(tstop - tstart))

        out_gpu = x_gpu._empty_like_me()
        tstart = time()
        x_gpu_to_cpu = np.asarray(x_gpu)
        out_cpu = b * np.asarray(y_gpu)
        out_cpu += a * x_gpu_to_cpu
        out_gpu[:] = out_cpu
        tstop = time()
        print('CPU time, with copy: {:.5}'.format(tstop - tstart))

        print('')


# %%

print_times_scal()
print_times_axpy()
print_times_axpby()
print_times_lico()
