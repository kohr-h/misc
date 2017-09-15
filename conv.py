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


class Convolution(Operator):

    """Discretized continuous or fully discrete convolution."""

    def __init__(self, domain, kernel, range=None, axis=None, impl='fft',
                 **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `TensorSpace`
            Space on which the convolution is defined. If ``domain`` is
            a `DiscreteLp`, it must be uniformly discretized. If it is
            not a `DiscreteLp`, the convolution is fully discrete, and
            the kernel is treated as array-like.
        kernel : `DiscreteLpElement` or array-like
            The kernel with which input elements are convolved. Its shape
            can be at most ``domain.shape``.

            - `DiscreteLpElement`: If ``domain`` is a `DiscreteLp`,
              the ``kernel.space.domain`` determines the range shift,
              see Notes for details.
              The ``kernel.space.cell_sides`` must coincide with those
              of ``domain``. This also implies that ``kernel.space``
              must be uniformly discretized.

              Otherwise, ``kernel`` will be treated as an array.

            - array-like: If ``domain`` is a `DiscreteLp`, the kernel
              will be treated as an element in a zero-centered space with
              the same cell sides as ``domain``.

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
        """
        from builtins import range as builtin_range
        assert isinstance(domain, TensorSpace)
        cont_dom = isinstance(domain, DiscreteLp)
        cont_ker = isinstance(kernel, DiscreteLpElement)

        # Determine range and kernel space if necessary
        if cont_dom:
            if cont_ker:
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

        else:
            if not isinstance(kernel, Tensor):
                kernel = np.asarray(kernel)
                if kernel.flags.f_contiguous and not kernel.flags.c_contiguous:
                    order = 'F'
                else:
                    order = 'C'
                ker_space = tensor_space(kernel.shape, kernel.dtype, order)

            if range is None:
                range = domain

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

