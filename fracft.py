from __future__ import print_function, division, absolute_import

import numpy as np


__all__ = ('fractional_ft',)


# @profile  # for the line profiler
def fractional_ft(x, alpha, axis=0, out=None, padded_len=None, impl='numpy',
                  **kwargs):
    """Compute the fractional FT of x with parameter alpha.

    The fractional FT of ``x`` with parameter ``alpha`` is defined as ::

        G[k] = sum_j( x[j] * exp(-1j * 2*pi * j*k*alpha) )

    If ``alpha == 1 / len(x)``, this is the usual DFT. The sum can be
    written as a circular convolution of length ``2 * p``::

        G[k] = conj(z[k]) * sum_j (y[j] * z[k - j]), 0 <= k < len(x),

    where ``z[j] = exp(1j * pi * alpha * j**2)`` and
    ``y[j] = x[j] * conj(z[j])`` for ``0 <= j < len(x)`` and
    appropriately padded for ``len(x) <= j < 2 * p``. The parameter
    ``p`` can be chosen freely from the integers larger than or equal
    to ``len(x) - 1``.

    For higher dimensions, the described transform is applied in the given
    axis.

    Parameters
    ----------
    x : `array-like`
        Input array to be transformed.
    alpha : float or `array-like`
        Parameter in the complex exponential of the transform. An
        array-like must be broadcastable with ``x``.
    axis : int, optional
        Axis of ``x`` in which to compute the transform.
    out : `numpy.ndarray`, optional
        Array to store the values in.
    padded_len : int, optional
        Length of the padded array in the given axis. By default,
        ``padded_len = 2 * (n - 1)`` is chosen, where
        ``n == x.shape[axis]``. This is the smallest possible choice.
        Selecting a power of 2 may speed up the computation.
    impl : {'numpy', 'pyfftw', 'cufft'}
        Backend for computing the FFTs. Currently only ``'numpy'`` is
        supported.
    precomp_zbar : `array-like`, optional
        Array of precomputed factors ``zbar[j] = exp(-1j * pi * alpha * j**2)``
        as used in the transform (they are the complex conjugates of the
        ``z`` factors). Its shape must be broadcastable with
        ``x``, apart from ``axis``, where it must be at least as long
        as ``x``. Values at indices beyond the length of ``x`` in ``axis``
        are ignored.
    precomp_zhat : `array-like`, optional
        Array of precomputed factors (one per axis), which are the Fourier
        transforms of the factors ``z``.

    Returns
    -------
    out : `numpy.ndarray`
        The fractional FT of ``x``. The returned array has the same
        shape as ``x`` (padded values are discarded). If ``out`` was
        given, the returned object is a reference to it.
    precomp_zbar : `numpy.ndarray`
        The precomputed values of the complex conjugate of ``z``.
        If ``precomp_zbar`` was given as Numpy array, the returned object
        is a reference to it.
    precomp_zhat : tuple of `numpy.ndarray`
        The precomputed values of the DFT of ``z``. If ``precomp_zhat``
        was given as a Numpy array, the returned object is a reference
        to it.
    """
    # --- Process input parameters --- #

    # x
    x = np.asarray(x)
    if x.ndim < 1:
        raise ValueError('`x` must be at least 1-dimensional')
    # TODO: keep real data type and use half-complex in that case
    cplx_dtype = np.result_type(x.dtype, np.complex64)
    x = x.astype(cplx_dtype)
    order = 'F' if x.flags.f_contiguous and not x.flags.c_contiguous else 'C'

    # axis
    axis, axis_in = int(axis), axis
    axis = x.ndim + axis if axis < 0 else axis
    if not 0 <= axis < x.ndim:
        raise ValueError('`axis` {} out of the valid range {}, ..., {}'
                         ''.format(axis_in, -x.ndim, x.ndim - 1))

    # alpha
    real_dtype = np.empty(0, dtype=cplx_dtype).real.dtype
    alpha = np.array(alpha, dtype=real_dtype, copy=False, order=order)

    # padded_len
    if padded_len is None:
        padded_len = 2 * (x.shape[axis] - 1)
    else:
        padded_len, padded_len_in = int(padded_len), padded_len
        if padded_len_in < 2 * (x.shape[axis] - 1):
            raise ValueError(
                '`padded_len` must be at least {} for axis {} with length {}, '
                'got {}.'.format(2 * (x.shape[axis] - 1), axis, x.shape[axis],
                                 padded_len_in))
        if padded_len_in % 2:
            raise ValueError('`padded_len` must be even, got {}.'
                             ''.format(padded_len_in))

    # impl
    impl, impl_in = str(impl).lower(), impl
    if impl not in ('numpy', 'pyfftw', 'cufft'):
        raise ValueError('`impl` {!r} not understood'.format(impl_in))

    if impl != 'numpy':
        raise NotImplementedError('`impl` {!r} not supported yet'.format(impl))

    # precomp_z
    precomp_z = kwargs.pop('precomp_z', None)
    if precomp_z is None:
        # Initialize the precomputed z values. These are
        # exp(1j * pi * alpha * j**2) for 0 <= j < n
        js_sq = np.arange(x.shape[axis]) ** 2
        bcast_slc = [None] * x.ndim
        bcast_slc[axis] = slice(None)
        js_sq = js_sq[bcast_slc]
        precomp_zbar = np.exp((-1j * np.pi * js_sq) * alpha)

    precomp_zhat = kwargs.pop('precomp_zhat', None)
    if precomp_zhat is None:
        # Initialize the padded FT of the precomputed z values. These are
        # o exp(1j * pi * alpha * j**2) for 0 <= j < len(x)
        # o exp(1j * pi * alpha * (2*p - j)**2) for 2*p - m <= j < 2*p
        # o 0, otherwise
        # o followed by a discrete FT.
        # Here, 2*p refers to the even padded length of the arrays.
        js_sq = np.arange(x.shape[axis]) ** 2
        bcast_slc = [None] * x.ndim
        bcast_slc[axis] = slice(None)
        js_sq = js_sq[bcast_slc]

        shape = list(np.broadcast(js_sq, x, alpha).shape)
        shape[axis] = padded_len
        precomp_zhat = np.zeros(shape, dtype=cplx_dtype, order=order)

        # Lower part in `axis` (0 <= j < len(x) above)
        lower_slc = [slice(None)] * x.ndim
        lower_slc[axis] = slice(None, x.shape[axis])
        precomp_zhat[lower_slc] = np.exp((1j * np.pi * js_sq) * alpha)

        # Upper part (2*p - m <= j < 2*p above), gained by mirroring the
        # lower part from index 1 on
        upper_slc = [slice(None)] * x.ndim
        upper_slc[axis] = slice(-x.shape[axis] + 1, None)
        lower_mirr_slc = [slice(None)] * x.ndim
        # TODO: not sure if this slicing is correct (should maybe cover less?)
        lower_mirr_slc[axis] = slice(x.shape[axis] - 1, 0, -1)
        precomp_zhat[upper_slc] = precomp_zhat[lower_mirr_slc]

        if impl == 'numpy':
            precomp_zhat = np.fft.fft(precomp_zhat, axis=axis)
        else:
            assert False

    else:
        precomp_zhat = np.asarray(precomp_zhat, dtype=cplx_dtype, order=order)

    if out is None:
        out = np.empty_like(x)

    # Now the actual computation. First the input array x needs to be padded
    # with zeros up to padded_len (in a new array), and multiplied by the
    # z factors.
    shape = list(np.broadcast(x, alpha).shape)
    shape[axis] = padded_len
    x_part_slc = [slice(None)] * x.ndim
    x_part_slc[axis] = slice(None, x.shape[axis])
    y = np.zeros(shape, dtype=cplx_dtype, order=order)
    y[x_part_slc] = x
    y[x_part_slc] *= precomp_zbar[x_part_slc]

    # Now we convolve with the z values by performing FFT and multiplying
    # with the zhat values, then applying inverse FFT
    if impl == 'numpy':
        yhat = np.fft.fft(y, axis=axis)
    else:
        assert False

    yhat *= precomp_zhat

    if impl == 'numpy':
        y = np.fft.ifft(yhat, axis=axis)

    if out is None:
        out = y[x_part_slc]
    else:
        out[:] = y[x_part_slc]

    out *= precomp_zbar[x_part_slc]

    return out, precomp_zbar, precomp_zhat


def fracft_1d_direct(x, alpha):
    plen = 2 * (len(x) - 1)
    z = np.exp(1j * np.pi * np.arange(len(x)) ** 2 * alpha)
    zhat = np.zeros(plen, dtype=complex)
    zhat[:len(x)] = z
    zhat[-len(x) + 1:] = zhat[len(x) - 1: 0: -1]
    zhat = np.fft.fft(zhat)

    y = np.zeros(plen, dtype=complex)
    y[:len(x)] = x * z.conj()
    yhat = np.fft.fft(y)
    yhat *= zhat
    y = np.fft.ifft(yhat)
    y[:len(x)] *= z.conj()

    return y[:len(x)]


if __name__ == '__main__':
    x = np.zeros((2048, 500))
    alpha = 1. / 1024
    xhat = fractional_ft(x, alpha, axis=0)
