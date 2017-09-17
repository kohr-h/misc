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
from odl.util import is_real_dtype, is_floating_dtype


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

            **Important:**

            - The kernel must **always** have the same number of dimensions
              as ``domain``, even for convolutions along axes.
            - In axes where no convolution is performed, the shape of the
              kernel must either be 1 (broadcasting along these axes), or
              equal to the domain shape.

            See Examples for further clarification.

        range : `TensorSpace`, optional
            Space of output elements of the convolution. Must be of the
            same shape as ``domain``. If not given, the range is equal to
            ``domain.astype(result_dtype)``, where ``result_dtype`` is
            the data type of the convolution result. If ``impl='real'``,
            integer dtypes are preserved, while for ``impl='fft'``,
            the smallest possible floating-point type is chosen.
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
        - For ``impl='fft'``, the out-of-place variant is also faster since
          padding and un-padding the NumPy FFT backend does not support
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
        - The NumPy FFT backend always uses ``'float64'`` internally,
          so different data types will simply result in additional casting,
          not speedup or higher precision.

        Examples
        --------
        Convolve in all axes:

        >>> space = odl.rn((3, 3))
        >>> kernel = [[0, 0, 0],  # A discrete Dirac delta
        ...           [0, 1, 0],
        ...           [0, 0, 0]]
        >>> conv = DiscreteConvolution(space, kernel)
        >>> x = space.element([[1, 2, 3],
        ...                    [2, 4, 6],
        ...                    [-3, -6, -9]])
        >>> conv(x)
        rn((3, 3)).element(
            [[ 1.,  2.,  3.],
             [ 2.,  4.,  6.],
             [-3., -6., -9.]]
        )

        For even-sized kernels, the convolution is performed in a
        "backwards" manner, i.e., the lower indices are affected by
        implicit zero-padding:

        >>> kernel = [[1, 1],  # 2x2 blurring kernel
        ...           [1, 1]]
        >>> conv = DiscreteConvolution(space, kernel)
        >>> x = space.element([[1, 2, 3],
        ...                    [2, 4, 6],
        ...                    [-3, -6, -9]])
        >>> conv(x)
        rn((3, 3)).element(
            [[  1.,   3.,   5.],
             [  3.,   9.,  15.],
             [ -1.,  -3.,  -5.]]
        )

        Convolution in selected axes can be done either with broadcasting
        or with "stacked kernels":

        >>> kernel_1d = [1, -1]  # backward difference kernel
        >>> kernel = np.array(kernel_1d)[None, :]  # broadcasting in axis 0
        >>> conv = DiscreteConvolution(space, kernel, axis=1)
        >>> x = space.element([[1, 2, 3],
        ...                    [2, 4, 6],
        ...                    [-3, -6, -9]])
        >>> conv(x)
        rn((3, 3)).element(
            [[ 1.,  1.,  1.],
             [ 2.,  2.,  2.],
             [-3., -3., -3.]]
        )
        >>> kernel_stack = [[1, -1],  # separate kernel per row
        ...                 [2, -2],
        ...                 [3, -3]]
        >>> conv = DiscreteConvolution(space, kernel_stack, axis=1)
        >>> conv(x)
        rn((3, 3)).element(
            [[ 1.,  1.,  1.],
             [ 4.,  4.,  4.],
             [-9., -9., -9.]]
        )
        """
        from builtins import range as builtin_range
        assert isinstance(domain, TensorSpace)

        if not isinstance(kernel, Tensor):
            kernel = np.asarray(kernel)
            ker_space = tensor_space(kernel.shape, kernel.dtype, order='A')
            kernel = ker_space.element(kernel)

        if range is None:
            result_dtype = np.result_type(domain.dtype, kernel.dtype)
            if str(impl).lower() == 'fft':
                result_dtype = np.result_type(result_dtype, np.float16)
            range = domain.astype(result_dtype)

        super(DiscreteConvolution, self).__init__(domain, range, linear=True)

        self.__kernel = kernel

        if axis is None:
            self.__axes = tuple(builtin_range(self.domain.ndim))
        else:
            try:
                iter(axis)
            except TypeError:
                self.__axes = (int(axis),)
            else:
                self.__axes = tuple(int(ax) for ax in axis)

        assert all(kernel.shape[i] == 1 or
                   kernel.shape[i] == self.domain.shape[i]
                   for i in builtin_range(self.domain.ndim)
                   if i not in self.axes)

        self.__impl = str(impl).lower()
        if self.impl == 'real':
            assert self.axes == tuple(builtin_range(self.domain.ndim))
            self.__real_impl = 'scipy'
            self.__fft_impl = None
        elif self.impl == 'fft':
            self.__real_impl = None
            self.__fft_impl = 'pyfftw' if PYFFTW_AVAILABLE else 'numpy'

        padding = kwargs.pop('padding', None)
        padded_shape = kwargs.pop('padded_shape', None)
        assert padding is None or padded_shape is None

        if padding is None:
            full_padding = np.minimum(np.array(self.kernel.shape) - 1, 64)
            padding = [full_padding[i] if i in self.axes else 0
                       for i in builtin_range(self.domain.ndim)]
        else:
            try:
                iter(padding)
            except TypeError:
                padding = [int(padding) if i in self.axes else 0
                           for i in builtin_range(self.domain.ndim)]
            else:
                padding = [int(p) for p in padding]
                if len(padding) == len(self.axes):
                    padding_lst = [0] * self.domain.ndim
                    for ax, pad in zip(self.axes, padding):
                        padding_lst[ax] = pad
                    padding = padding_lst

        if padded_shape is None:
            padded_shape = tuple(np.array(self.domain.shape) + padding)

        self.__padded_shape = padded_shape

        self.__cache_kernel_ft = bool(kwargs.pop('cache_kernel_ft', False))
        self._kernel_ft = None

    @property
    def kernel(self):
        """The `Tensor` used as kernel in the convolution."""
        return self.__kernel

    @property
    def axes(self):
        """The dimensions along which the convolution is taken."""
        return self.__axes

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

    @property
    def cache_kernel_ft(self):
        """If ``True``, the kernel FT is cached for later reuse."""
        return self.__cache_kernel_ft

    def _call(self, x, out=None):
        """Perform convolution of ``f`` with `kernel`."""
        if self.impl == 'real' and self.real_impl == 'scipy':
            return self._call_scipy_convolve(x, out)
        elif self.impl == 'fft' and self.fft_impl == 'numpy':
            return self._call_numpy_fft(x, out)
        elif self.impl == 'fft' and self.fft_impl == 'pyfftw':
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
        # Use real-to-complex FFT if possible, it's faster
        if (is_real_dtype(self.kernel.dtype) and
                is_real_dtype(self.domain.dtype)):
            fft = np.fft.rfftn
            ifft = np.fft.irfftn
        else:
            fft = np.fft.fftn
            ifft = np.fft.ifftn

        # Prepare kernel, preserving length-1 axes for broadcasting
        ker_padded_shp = [1 if self.kernel.shape[i] == 1
                          else self.padded_shape[i]
                          for i in range(self.domain.ndim)]
        kernel_prep = prepare_for_fft(self.kernel, ker_padded_shp, self.axes)

        # Pad the input with zeros
        paddings = []
        for i in range(self.domain.ndim):
            diff = self.padded_shape[i] - x.shape[i]
            left = diff // 2
            right = diff - left
            paddings.append((left, right))

        x_prep = np.pad(x, paddings, 'constant')

        # Perform FFTs of x and kernel (or retrieve from cache)
        x_ft = fft(x_prep, axes=self.axes)

        if self._kernel_ft is not None:
            kernel_ft = self._kernel_ft
        else:
            kernel_ft = fft(kernel_prep, axes=self.axes)
            if self.cache_kernel_ft:
                self._kernel_ft = kernel_ft

        # Multiply x_ft with kernel_ft and transform back. Note that
        # x_ft and kernel_ft have dtype 'float64' since that's what
        # numpy.fft does.
        x_ft *= kernel_ft
        # irfft needs an explicit shape, otherwise the result shape may not
        # be the same as the original one
        s = [x_prep.shape[i]
             for i in range(self.domain.ndim) if i in self.axes]
        ifft_x = ifft(x_ft, axes=self.axes, s=s)

        # Unpad to get the "relevant" part
        slc = [slice(l, n - r) for (l, r), n in zip(paddings, x_prep.shape)]
        if out is None:
            out = ifft_x[slc]
        else:
            out[:] = ifft_x[slc]

        return out

    def _call_pyfftw(self, x, out=None):
        """Perform FFT-based convolution using the pyfftw backend."""
        import multiprocessing
        import pyfftw

        # Prepare kernel, preserving length-1 axes for broadcasting
        if is_floating_dtype(self.kernel.dtype):
            kernel = self.kernel
        else:
            flt_dtype = np.result_type(self.kernel.dtype, np.float16)
            kernel = np.asarray(self.kernel, dtype=flt_dtype)

        ker_padded_shp = [1 if self.kernel.shape[i] == 1
                          else self.padded_shape[i]
                          for i in range(self.domain.ndim)]
        kernel_prep = prepare_for_fft(kernel, ker_padded_shp, self.axes)
        kernel = None  # can be gc'ed

        # Pad the input with zeros
        paddings = []
        for i in range(self.domain.ndim):
            diff = self.padded_shape[i] - x.shape[i]
            left = diff // 2
            right = diff - left
            paddings.append((left, right))

        # TODO: order
        x_prep = np.pad(x, paddings, 'constant')
        x_prep_shape = x_prep.shape

        # Real-to-halfcomplex only if both domain and kernel are eligible
        use_halfcx = (is_real_dtype(self.domain.dtype) and
                      is_real_dtype(self.kernel.dtype))

        def fft_out_array(arr, use_halfcx):
            """Make an output array for FFTW with suitable dtype and shape."""
            ft_dtype = np.result_type(arr.dtype, 1j)
            ft_shape = list(arr.shape)
            if use_halfcx:
                ft_shape[self.axes[-1]] = ft_shape[self.axes[-1]] // 2 + 1
            return np.empty(ft_shape, ft_dtype)

        # Perform FFT of x. Use 'FFTW_ESTIMATE', since other options destroy
        # the input and would require a copy.
        x_ft = fft_out_array(x_prep, use_halfcx)
        if not use_halfcx and x_ft.dtype != x_prep.dtype:
            # Need to perform C2C transform, hence a cast
            x_prep = x_prep.astype(x_ft.dtype)

        plan_x = pyfftw.FFTW(x_prep, x_ft, axes=self.axes,
                             direction='FFTW_FORWARD',
                             flags=['FFTW_ESTIMATE'],
                             threads=multiprocessing.cpu_count())
        plan_x(x_prep, x_ft)
        plan_x = None  # can be gc'ed
        x_prep = None

        # Perform FFT of kernel if necessary
        if self._kernel_ft is not None:
            kernel_ft = self._kernel_ft
        else:
            kernel_ft = fft_out_array(kernel_prep, use_halfcx)
            if not use_halfcx and kernel_ft.dtype != kernel_prep.dtype:
                # Need to perform C2C transform, hence a cast
                kernel_prep = kernel_prep.astype(kernel_ft.dtype)

            plan_kernel = pyfftw.FFTW(kernel_prep, kernel_ft, axes=self.axes,
                                      direction='FFTW_FORWARD',
                                      flags=['FFTW_ESTIMATE'],
                                      threads=multiprocessing.cpu_count())
            plan_kernel(kernel_prep, kernel_ft)
            plan_kernel = None  # can be gc'ed
            kernel_prep = None

        # Multiply x_ft with kernel_ft and transform back. Some care
        # is required with respect to dtypes, in particular when
        # x_ft.dtype < kernel_ft.dtype.
        if x_ft.dtype < kernel_ft.dtype:
            x_ft = x_ft * kernel_ft
        else:
            x_ft *= kernel_ft

        # Perform inverse FFT
        x_ift_dtype = np.empty(0, dtype=x_ft.dtype).real.dtype
        x_ift = np.empty(x_prep_shape, x_ift_dtype)
        plan_ift = pyfftw.FFTW(x_ft, x_ift, axes=self.axes,
                               direction='FFTW_BACKWARD',
                               flags=['FFTW_ESTIMATE'],
                               threads=multiprocessing.cpu_count())

        plan_ift(x_ft, x_ift)
        x_ft = None  # can be gc'ed

        # Unpad to get the "relevant" part
        slc = [slice(l, n - r) for (l, r), n in zip(paddings, x_prep_shape)]
        if out is None:
            out = x_ift[slc]
        else:
            out[:] = x_ift[slc]

        return out


def prepare_for_fft(kernel, padded_shape, axes=None):
    """Return a kernel with desired shape with middle entry at index 0.

    This function applies the appropriate steps to prepare a kernel for
    FFT-based convolution. It first pads the kernel with zeros *to the
    right* up to ``padded_shape``, and then rolls the entries such that
    the old middle element, i.e., the one at ``(kernel.shape - 1) // 2``,
    lies at index 0.

    Parameters
    ----------
    kernel : array-like
        The kernel to be prepared for FFT convolution.
    padded_shape : sequence of ints
        The target shape to be reached by zero-padding.
    axes : sequence of ints, optional
        Dimensions in which to perform shifting. ``None`` means all axes.

    Returns
    -------
    prepared : `numpy.ndarray`
        The zero-padded and rolled kernel ready for FFT.

    Examples
    --------
    >>> kernel = np.array([[1, 2, 3],
    ...                    [4, 5, 6]])  # middle element is 2
    >>> prepare_for_fft(kernel, padded_shape=(4, 4))
    array([[2, 3, 0, 1],
           [5, 6, 0, 4],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    >>> prepare_for_fft(kernel, padded_shape=(5, 5))
    array([[2, 3, 0, 0, 1],
           [5, 6, 0, 0, 4],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    """
    kernel = np.asarray(kernel)
    if kernel.flags.f_contiguous and not kernel.flags.c_contiguous:
        order = 'F'
    else:
        order = 'C'

    padded = np.zeros(padded_shape, kernel.dtype, order)

    if axes is None:
        axes = list(range(kernel.ndim))

    if any(padded_shape[i] != kernel.shape[i] for i in range(kernel.ndim)
           if i not in axes):
        raise ValueError(
            '`padded_shape` can only differ from `kernel.shape` in `axes`; '
            'got `padded_shape={}`, `kernel.shape={}`, `axes={}`'
            ''.format(padded_shape, kernel.shape, axes))

    orig_slc = [slice(n) for n in kernel.shape]
    padded[orig_slc] = kernel
    # This shift makes sure that the middle element is shifted to index 0
    shift = [-((kernel.shape[i] - 1) // 2) if i in axes else 0
             for i in range(kernel.ndim)]
    return np.roll(padded, shift, axis=axes)


if __name__ == '__main__':
    from odl.util import run_doctests
    run_doctests()
