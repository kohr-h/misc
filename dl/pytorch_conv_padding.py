import numpy as np
import scipy.signal
import torch
from torch import autograd, nn
from odl.contrib import fom

np.random.seed(123)
arr = np.random.randint(0, 5, size=(5, 6)).astype(float)


def filter_image(image, fh, fv):
    """Reference filtering function using ``scipy.signal.convolve``."""
    fh, fv = np.asarray(fh), np.asarray(fv)
    image = scipy.signal.convolve(image, fh[:, None], mode='same')
    return scipy.signal.convolve(image, fv[None, :], mode='same')


fh = np.array([-1, 1], dtype=float)
fv = np.array([1, 1], dtype=float)

filtered_real = filter_image(arr, fh, fv)
filtered_fft = fom.util.filter_image_fft(arr, fh, fv)
print('real and fft equal?', np.allclose(filtered_real, filtered_fft))

x = autograd.Variable(torch.DoubleTensor(arr))
filt = autograd.Variable(
    torch.DoubleTensor(np.multiply.outer(fh[::-1], fv[::-1], dtype=float)))
filtered_torch = nn.functional.conv2d(x[None, None, ...],
                                      filt[None, None, ...], padding=1)
print(filtered_real)
print(filtered_torch[0, 0, :-1, :-1])

fh = np.repeat(fh, 2)
fv = np.repeat(fv, 2)

filtered_real = filter_image(arr, fh, fv)
filtered_fft = fom.util.filter_image_fft(arr, fh, fv)
print('real and fft equal?', np.allclose(filtered_real, filtered_fft))

x = autograd.Variable(torch.DoubleTensor(arr))
filt = autograd.Variable(
    torch.DoubleTensor(np.multiply.outer(fh[::-1], fv[::-1], dtype=float)))
filtered_torch = nn.functional.conv2d(x[None, None, ...],
                                      filt[None, None, ...], padding=3)
print(filtered_real)
print(filtered_torch[0, 0, 1:-2, 1:-2])
