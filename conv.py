# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division

import numpy as np

from odl.discr import DiscreteLp, DiscreteLpElement, uniform_discr_fromdiscr
from odl.operator import Operator
from odl.space import tensor_space
from odl.space.base_tensors import TensorSpace, Tensor
from odl.trafos.backends import PYFFTW_AVAILABLE


class Convolution(Operator):

    """Discretized continuous convolution with a given kernel."""

    def __init__(self, domain, kernel, range=None, axis=None, impl='fft',
                 **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscreteLp`
            Space on which the convolution is defined, must be uniformly
            discretized.
        kernel : `DiscreteLpElement` or array-like
            The kernel with which input elements are convolved. If a
            `DiscreteLpElement` is given, its domain of definition determines
            the range of the operator as shifted by
            ``kernel.space.mid_pt``. The ``kernel.space.cell_sizes`` must
            coincide with ``domain.cell_sizes``, except in axes with
            size 1.

            An array-like object is wrapped
            into an element of a `DiscreteLp` centered at zero, and using
            ``domain.cell_sizes``.

            In axes where ``kernel.shape`` is 1, broadcasting applies.

        range : `TensorSpace`, optional
            Space of output elements of the convolution. Must be of the
            same shape as ``domain``. If not given, the range is inferred
            from ``domain`` and ``kernel``, see Notes.
        axis : int or sequence of ints, optional
            Coordinate axis or axes in which to take the convolution.
            ``None`` means all input axes.
        impl : {'fft', 'real'}
            Implementation of the convolution as FFT-based or using
            real-space summation. The fastest available FFT backend is
            chosen automatically. Real space convolution is based on
            `scipy.signal.convolve`.

        Other Parameters
        ----------------
        padding : int or sequence of ints, optional
            Zero-padding used before Fourier transform in the FFT backend.
            Does not apply for ``impl='real'``. A sequence is applied per
            axis, with padding values corresponding to ``axis`` entries
            as provided.
            Default: ``min(kernel.shape - 1, 64)``
        padded_shape : sequence of ints, optional
            Apply zero-padding with this target shape. Cannot be used
            together with ``padding``.
        cache_kernel_ft : bool, optional
            If ``True``, store the Fourier transform of the kernel for
            later reuse.
            Default: ``False``

        Notes
        -----
        TODO describe shifting
        """
        from builtins import range as builtin_range
        assert isinstance(domain, DiscreteLp)

        # Determine range and kernel space if necessary
        if isinstance(kernel, DiscreteLpElement):
            assert all(np.isclose(domain.cell_sides[i],
                                  kernel.space.cell_sides[i])
                       for i in range(domain.ndim) if kernel.shape[i] != 1)
            ran_shift = (domain.partition.mid_pt +
                         kernel.space.partition.mid_pt)
        else:
            ran_shift = np.zeros(domain.ndim)
            kernel = np.asarray(kernel)
            ker_min = -domain.cell_sides / 2 * kernel.shape
            ker_space = uniform_discr_fromdiscr(
                domain, min_pt=ker_min, shape=kernel.shape)
            kernel = ker_space.element(kernel)

        if range is None:
            ran_min = domain.min_pt + ran_shift
            range = uniform_discr_fromdiscr(domain, min_pt=ran_min)

        super(Convolution, self).__init__(domain, range, linear=True)

        self.__kernel = kernel

        if axis is None:
            self.__axis = tuple(builtin_range(domain.ndim))
        else:
            try:
                iter(axis)
            except TypeError:
                self.__axis = (int(axis),)
            else:
                self.__axis = tuple(int(ax) for ax in axis)

        self.__impl = str(impl).lower()

        padding = kwargs.pop('padding', None)
        padded_shape = kwargs.pop('padded_shape', None)
        assert padding is None or padded_shape is None

        if padding is None:
            padding = tuple(np.minimum(np.array(self.kernel.shape) - 1, 64))
        else:
            try:
                iter(padding)
            except TypeError:
                padding = tuple(int(padding) if i in self.axis else 0
                                for i in builtin_range(self.domain.ndim))
            else:
                padding = tuple(int(p) for p in padding)
                if len(padding) == len(self.axis):
                    padding_lst = [0] * self.domain.ndim
                    for ax, pad in zip(self.axis, padding):
                        padding_lst[ax] = pad
                    padding = tuple(padding_lst)

        if padded_shape is None:
            padded_shape = tuple(np.array(self.domain.shape) + padding)

        self.__padded_shape = padded_shape

    @property
    def kernel(self):
        """The `Tensor` used as kernel in the convolution."""
        return self.__kernel

    @property
    def axis(self):
        """The axis or axes in which the convolution is taken."""
        return self.__axis

    @property
    def impl(self):
        """Implementation variant, ``'fft' or 'real'``."""
        return self.__impl

    @property
    def padded_shape(self):
        """Domain shape after padding for FFT-based convolution."""
        return self.__padded_shape


class DiscreteConvolution(Operator):

    """Fully discrete convolution with a given kernel."""

    def __init__(self, domain, kernel, range=None, axis=None, impl='fft',
                 **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `TensorSpace`
            Space on which the convolution is defined. If ``domain`` is
            a `DiscreteLp`, it must be uniformly discretized.
        kernel : array-like
            The kernel with which input elements are convolved. It must
            have the same number of dimensions as ``domain``, and its
            shape can be at most equal to ``domain.shape``. In axes
            with size 1, broadcasting is applied.
        range : `TensorSpace`, optional
            Space of output elements of the convolution. Must be of the
            same shape as ``domain``. If not given, the range is equal
            to ``domain``.
        axis : int or sequence of ints, optional
            Coordinate axis or axes in which to take the convolution.
            ``None`` means all input axes.
        impl : {'fft', 'real'}
            Implementation of the convolution as FFT-based or using
            direct summation. The fastest available FFT backend is
            chosen automatically. Real space convolution is based on
            `scipy.signal.convolve`.
            See Notes for further information on the backends.

        Other Parameters
        ----------------
        padding : int or sequence of ints, optional
            Zero-padding used before Fourier transform in the FFT backend.
            Does not apply for ``impl='real'``. A sequence is applied per
            axis, with padding values corresponding to ``axis`` entries
            as provided.
            Default: ``min(kernel.shape - 1, 64)``
        padded_shape : sequence of ints, optional
            Apply zero-padding with this target shape. Cannot be used
            together with ``padding``.
        cache_kernel_ft : bool, optional
            If ``True``, store the Fourier transform of the kernel for
            later reuse.
            Default: ``False``

        Notes
        -----
        - For ``impl='real'``, the out-of-place call (no ``out`` parameter)
          is faster since the backend does not support writing to an
          existing array.
        - For ``impl='fft'``, the NumPy FFT backend does not support
          ``out`` arrays either and will thus also result in the
          out-of-place variant being faster. If the ``pyfftw`` backend is
          used, in-place evaluation is faster.
        - ``scipy.convolve`` does not support an ``axis`` parameter.
          However, a convolution along axes with a lower-dimensional
          kernel can be achieved by adding empty dimensions. For example,
          to convolve along axis 0 we can do the following ::

              ker_1d = np.array([-1.0, 1.0])
              ker_axis0 = ker_1d[:, None]
              conv = DiscreteConvolution(space_2d, ker_axis0, impl='real')

          Not possible with this approach is here a convolution with a
          *different* kernel in each column.
        """
        from builtins import range as builtin_range
        assert isinstance(domain, TensorSpace)

        if not isinstance(kernel, Tensor):
            kernel = np.asarray(kernel)
            ker_space = tensor_space(kernel.shape, kernel.dtype, order='A')
            kernel = ker_space.element(kernel)

        if range is None:
            range = domain

        super(DiscreteConvolution, self).__init__(domain, range, linear=True)

        self.__kernel = kernel

        if axis is None:
            self.__axis = tuple(builtin_range(self.domain.ndim))
        else:
            try:
                iter(axis)
            except TypeError:
                self.__axis = (int(axis),)
            else:
                self.__axis = tuple(int(ax) for ax in axis)

        self.__impl = str(impl).lower()
        if self.impl == 'real':
            assert self.axis == tuple(builtin_range(self.domain.ndim))
            self.__real_impl = 'scipy'
            self.__fft_impl = None
        elif self.impl == 'fft':
            self.__real_impl = None
            self.__fft_impl = 'pyfftw' if PYFFTW_AVAILABLE else 'numpy'

        padding = kwargs.pop('padding', None)
        padded_shape = kwargs.pop('padded_shape', None)
        assert padding is None or padded_shape is None

        if padding is None:
            padding = tuple(np.minimum(np.array(self.kernel.shape) - 1, 64))
        else:
            try:
                iter(padding)
            except TypeError:
                padding = tuple(int(padding) if i in self.axis else 0
                                for i in builtin_range(self.domain.ndim))
            else:
                padding = tuple(int(p) for p in padding)
                if len(padding) == len(self.axis):
                    padding_lst = [0] * self.domain.ndim
                    for ax, pad in zip(self.axis, padding):
                        padding_lst[ax] = pad
                    padding = tuple(padding_lst)

        if padded_shape is None:
            padded_shape = tuple(np.array(self.domain.shape) + padding)

        self.__padded_shape = padded_shape

    @property
    def kernel(self):
        """The `Tensor` used as kernel in the convolution."""
        return self.__kernel

    @property
    def axis(self):
        """The axis or axes in which the convolution is taken."""
        return self.__axis

    @property
    def impl(self):
        """Implementation variant, ``'fft' or 'real'``."""
        return self.__impl

    @property
    def real_impl(self):
        """Backend for real-space conv., or ``None`` if not applicable."""
        return self.__real_impl

    @property
    def fft_impl(self):
        """Backend used for FFTs, or ``None`` if not applicable."""
        return self.__fft_impl

    @property
    def padded_shape(self):
        """Domain shape after padding for FFT-based convolution."""
        return self.__padded_shape

    def _call(self, x, out=None):
        """Perform convolution of ``f`` with `kernel`."""
        if self.impl == 'real' and self.call_real == 'scipy':
            return self._call_scipy(x, out)
        elif self.impl == 'fft' and self.fft_impl == 'numpy':
            return self._call_numpy_fft(x, out)
        elif self.impl == 'fft' and self.fft_impl == 'pfftw':
            return self._call_pyfftw(x, out)
        else:
            raise RuntimeError('bad `impl` {!r} or `fft_impl` {!r}'
                               ''.format(self.impl, self.fft_impl))

    def _call_scipy_convolve(self, x, out=None):
        """Perform real-space convolution using ``scipy.signal.convolve``."""
        import scipy.signal

        conv = scipy.signal.convolve(x, self.kernel, mode='same',
                                     method='direct')
        if out is None:
            out = conv
        else:
            out[:] = conv
        return out

    def _call_numpy_fft(self, x, out=None):
        """Perform FFT-based convolution using NumPy's backend."""



def kernel_padding_lengths(old_len, new_len):
    """Return left and right kernel padding sizes.

    When padding the kernel, the middle element, i.e., the one at index
    ``(old_len - 1) // 2``, has to be located at the new middle
    ``(new_len - 1) // 2`` after padding. This is achieved by adding
    ``(new_len - 1) // 2 - (old_len - 1) // 2`` zeros to the left, and
    enough zeros to the right to reach the desired length.

    Parameters
    ----------
    old_len, new_len : int
        Kernel size before and after padding, respectively.

    Returns
    -------
    num_left, num_right : int
        Number of zeros that should be added to the left and right.
    """
    num_left = (new_len - 1) // 2 - (old_len - 1) // 2
    num_right = new_len - old_len - num_left
    return num_left, num_right


def padded_kernel(kernel, padded_shape):
    """Zero-pad the kernel such that the middle entry is preserved.

    See ``kernel_padding_lengths`` for an explanation.

    Parameters
    ----------
    kernel : array-like
        The kernel to be padded.
    padded_shape : sequence of ints
        The target shape to be reached by zero-padding.

    Returns
    -------
    padded_kernel : `numpy.ndarray`
        The kernel padded with zeros, such that the middle element remains
        the same.
    """
    kernel = np.asarray(kernel)
    inner_slc = []
    for old_len, new_len in zip(kernel.shape, padded_shape):
        nl, nr = kernel_padding_lengths(old_len, new_len)
        inner_slc.append(slice(nl, new_len - nr))  # avoid -0

    if kernel.flags.f_contiguous and not kernel.flags.c_contiguous:
        order = 'F'
    else:
        order = 'C'

    padded = np.zeros(padded_shape, kernel.dtype, order)
    padded[inner_slc] = kernel
    return padded


def fftshift_kernel(kernel):
    """Perform an FFTshift-like operation on the kernel."""
    kernel = np.asarray(kernel)
    for i, n in enumerate(kernel.shape):
        slc_l = [slice(None)] * kernel.ndim
        slc_r = [slice(None)] * kernel.ndim
        slc_l[i] = slice((n - 1) // 2, None)
        slc_r[i] = slice((n - 1) // 2)
        kernel = np.concatenate([kernel[slc_l], kernel[slc_r]], axis=i)

    return kernel


def padded_kernel2(kernel, padded_shape):
    kernel = np.asarray(kernel)
    if kernel.flags.f_contiguous and not kernel.flags.c_contiguous:
        order = 'F'
    else:
        order = 'C'

    padded = np.zeros(padded_shape, kernel.dtype, order)

    orig_slc = [slice(n) for n in kernel.shape]
    padded[orig_slc] = kernel
    # This shift makes sure that the middle element is shifted to index 0
    shift = [-(n - 1) // 2 + 1 for n in kernel.shape]
    return np.roll(padded, shift, axis=range(kernel.ndim))


def dump():
    # Pad image with zeros
    if padding:
        image_padded = np.pad(image, padding, mode='constant')
    else:
        image_padded = image.copy() if impl == 'pyfftw' else image


    # Pad the filters
    def padded_filter(filt, n_new):
        """Return padded filter with new length.

        The filter is padded with zeros such that the middle element of
        the padded filter, i.e., the one with index
        ``(len(filt) - 1) // 2``, is the same as in the original filter.
        This is achieved by adding
        ``(n_new - 1) // 2 - (n_old - 1) // 2`` zeros to the left, and
        enough zeros to the right to reach the desired length.
        """
        n_old = len(filt)
        n_left = (n_new - 1) // 2 - (n_old - 1) // 2
        n_right = n_new - n_old - n_left
        left = np.zeros(n_left, dtype=filt.dtype)
        right = np.zeros(n_right, dtype=filt.dtype)
        return np.concatenate([left, filt, right])


    fh = np.asarray(fh).astype(image.dtype)
    if fh.ndim != 1:
        raise ValueError('`fh` must be one-dimensional')
    elif fh.size == 0:
        raise ValueError('`fh` cannot have size 0')
    elif fh.size > image.shape[0]:
        raise ValueError('`fh` can be at most `image.shape[0]`, got '
                         '{} > {}'.format(fh.size, image.shape[0]))
    else:
        fh = padded_filter(fh, image_padded.shape[0])


    fv = np.asarray(fv).astype(image.dtype)
    if fv.ndim != 1:
        raise ValueError('`fv` must be one-dimensional')
    elif fv.size == 0:
        raise ValueError('`fv` cannot have size 0')
    elif fv.size > image.shape[0]:
        raise ValueError('`fv` can be at most `image.shape[1]`, got '
                         '{} > {}'.format(fv.size, image.shape[1]))
    else:
        fv = padded_filter(fv, image_padded.shape[1])


    # We also need to perform a kind of FFTshift on the filters, namely
    # `[h_m, h_{m+1},..., h_{n-1}, h_0, ..., h_{m-1}]`, where
    # `m = (n - 1) // 2`. This is neither FFTshift nor IFFTshift exactly,
    # but easily written by hand.
    mid = (len(fh) - 1) // 2
    fh = np.concatenate([fh[mid:], fh[:mid]])
    mid = (len(fv) - 1) // 2
    fv = np.concatenate([fv[mid:], fv[:mid]])


    # Perform the multiplication in Fourier space
    if impl == 'numpy':
        image_ft = np.fft.rfftn(image_padded)
        fh_ft = np.fft.fft(fh)
        fv_ft = np.fft.rfft(fv)


        image_ft *= fh_ft[:, None]
        image_ft *= fv_ft[None, :]
        # Important to specify the shape since `irfftn` cannot know the
        # original shape
        conv = np.fft.irfftn(image_ft, s=image_padded.shape)
        if conv.dtype != image.dtype:
            conv = conv.astype(image.dtype)
