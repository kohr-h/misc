from __future__ import print_function, division, absolute_import

import numpy as np


__all__ = ('fractional_ft',)


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
        Length of the padded arrays in the given axis. By default,
        ``padded_len = 2 * (n - 1)`` is chosen, where
        ``n == x.shape[axis]``. This is the smallest possible choice.
        Selecting a power of 2 may speed up the computation.
    precomp_z : sequence of `array-like`, optional
        Arrays of precomputed factors (one per axis)
        ``z[j] = exp(1j * pi * alpha * j**2)`` as used in the
        transform. Their lengths must be at least the length of ``x``
        in the corresponding axes. Values at indices beyond the length
        of ``x`` in the respective axis are ignored.
    precomp_zhat : sequence of `array-like`, optional
        Arrays of precomputed factors (one per axis), which are the Fourier
        transforms of the factors ``z``.

    Returns
    -------
    out : `numpy.ndarray`
        The fractional FT of ``x``. The returned array has the same
        shape as ``x`` (padded values are discarded). If ``out`` was
        given, the returned object is a reference to it.
    precomp_z : tuple of `numpy.ndarray`
        The precomputed values of the DFT of ``z``. If ``precomp_z``
        was given as a tuple of ndarray, the returned object is a
        reference to it.
    precomp_zhat : tuple of `numpy.ndarray`
        The precomputed values of the DFT of ``zhat``. If ``precomp_zhat``
        was given as a tuple of ndarray, the returned object is a
        reference to it.
    """
    # --- Process input parameters --- #

    x = np.asarray(x)
    # TODO: keep real data type and use half-complex in that case
    cplx_dtype = np.result_type(x.dtype, np.complex64)
    x = x.astype(cplx_dtype)

    # axes
    if axes is None:
        axes = list(range(x.ndim))
    axes = [int(ax) for ax in axes]

    # alpha
    real_dtype = np.empty(0, dtype=cplx_dtype).real.dtype
    alpha = [np.array(a, dtype=real_dtype, copy=False) for a in alpha]

    if len(alpha) != len(axes):
        raise ValueError('lengths of `axes` and `alpha` do not match: '
                         '{} != {}'.format(len(alpha), len(axes)))

    # padded_len
    if padded_len is None:
        padded_len = [2 * (x.shape[ax] - 1) for ax in axes]
    else:
        padded_len = [int(plen) for plen in padded_len]

    if len(padded_len) != len(axes):
        raise ValueError('lengths of `axes` and `padded_len` do not match: '
                         '{} != {}'.format(len(padded_len), len(axes)))

    for ax, plen in zip(axes, padded_len):
        if plen < 2 * (x.shape[ax] - 1):
            raise ValueError('`padded_len` entry for axis {} must be at least '
                             '{}, got {}.'
                             ''.format(ax, 2 * (x.shape[ax] - 1), plen))
        if plen % 2:
            raise ValueError('`padded_len` entry for axis {} must be even, '
                             'got {}.'.format(ax, plen))

    # precomp_z
    precomp_z = kwargs.pop('precomp_z', None)
    if precomp_z is None:
        precomp_z = []
        for ax, a in zip(axes, alpha):
            # Initialize the precomputed z values. These are
            # exp(1j * pi * alpha * j**2) for 0 <= j < n
            arr = np.exp((1j * np.pi * np.arange(x.shape[ax]) ** 2) * a)
            precomp_z.append(arr)

    precomp_zhat = kwargs.pop('precomp_zhat', None)
    if precomp_zhat is None:
        for n, a, z in zip(shape, alpha, precomp_z):
            pass
        # Initialize the padded FT of the precomputed z values. These are
        # o exp(1j * pi * alpha * j**2) for 0 <= j < len(x)
        # o exp(1j * pi * alpha * (2*p - j)**2) for 2*p - m <= j < 2*p
        # o 0, otherwise
        # o followed by a discrete FT.
        # Here, 2*p refers to the even padded length of the arrays.
#        precomp_zhat = np.empty(padded_len, dtype='complex')
#        precomp_zhat[:len(x)] = precomp_z[:len(x)]
#        precomp_zhat[-len(x) + 1:] = precomp_z[1:len(x)][::-1]
#        # Here we get a copy, no way around since fft has no in-place method
#        precomp_zhat = np.fft.fft(precomp_zhat)
    else:
        precomp_zhat = np.asarray(precomp_zhat, dtype='complex')
        if precomp_zhat.ndim != 1:
            raise ValueError('precomp_zhat has {} dimensions, expected 1.'
                             ''.format(precomp_zhat.ndim))
        if len(precomp_zhat) != padded_len:
            raise ValueError('precomp_zhat has length {}, expected {}.'
                             ''.format(len(precomp_zhat), padded_len))

    # TODO: axis order
    if out is None:
        out = np.empty_like(x)
    else:
        if not isinstance(out, np.ndarray):
            raise TypeError('out is not a numpy.ndarray instance.'.format(out))
        if out.shape != x.shape:
            raise ValueError('out has shape {}, expected {}.'
                             ''.format(out.shape, x.shape))
        # TODO: adapt this once other dtypes are considered
        if out.dtype != x.dtype:
            raise ValueError('out has dtype {}, expected {}.'
                             ''.format(out.dtype, x.dtype))

    # Now the actual computation. First the input array x needs to be padded
    # with zeros up to padded_len (in a new array).

    y = np.zeros(padded_len, dtype='complex')
    y[:len(x)] = x
    y[:len(x)] *= precomp_z[:len(x)]
    yhat = np.fft.fft(y)
    yhat *= precomp_zhat
    y = np.fft.ifft(yhat)
    out[:] = y[:len(x)]
    out *= precomp_z[:len(x)].conj()

    return out, precomp_z, precomp_zhat


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
