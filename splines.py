#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 23:20:11 2017

@author: hkohr
"""

import mako
import numpy as np
import pygpu
from pygpu.tools import ScalarArg, ArrayArg
ctx = pygpu.init('cuda')
pygpu.set_default_context(ctx)
DTYPE_TO_CTYPE = {v: k
                  for k, v in pygpu.dtypes.NAME_TO_DTYPE.items()
                  if k.startswith('ga')}


# --- Kernel code definition --- #

kernel_tpl = mako.template.Template("""
/*************************************************************************/
/*
bspline3_norm_coord(x)

B-spline of order 3 in normalized coordinates (grid step 1).

Parameters
----------
x : float
    Evaluation point.

Returns
-------
spline_val : float
    B-spline value at x.
*/

WITHIN_KERNEL ${float}
bspline3_norm_coord(${float} x) {
    const ${float} two_m_x = 2.0 - x;
    if (x < 1.0)
        return -0.5 * x * x * two_m_x + 2.0 / 3.0 * x;
    else if (x < 2.0)
        return two_m_x * two_m_x * two_m_x / 6.0;
    else
        return 0.0;
}

/*************************************************************************/
/*
bspline_weights(lam, &a0, &a1, &a2, &a3)

Compute the filter weights of a B-spline at cell fraction lam.

Parameters
----------
lam : float
    Normalized position (0 <= lam < 1) in a cell where the B-spline
    should be evaluated.
&a0, &a1, &a2, &a3 : float_ptr
    Pointers to float values where the weights should be stored.

Returns
-------
None

*/
WITHIN_KERNEL void
bspline_weights(${float} lam,
                ${float} *a0,
                ${float} *a1,
                ${float} *a2,
                ${float} *a3
                ){
    const ${float} one_m_lam = 1.0 - lam;
    const ${float} lam_sq = lam * lam;
    const ${float} one_m_lam_sq = one_m_lam * one_m_lam;

    *a0 = one_m_lam_sq * one_m_lam / 6.0;
    *a1 = - 0.5 * lam_sq * (2.0 - lam) + 2.0 / 3.0;
    *a2 = - 0.5 * one_m_lam_sq * lam + 2.0 / 3.0 ;
    *a3 =  lam_sq * lam / 6.0;
    return;
}

/*************************************************************************/
/*
eval_bspline(x, gmin, gstep, gn)

Evaluate a 3rd order interpolating B-spline for the given grid points at
the point x.

Parameters
----------
x : float
    Point in which to evaluate the spline.
gmin : float
    Minimum grid point.
gstep : positive float
    Step between two consecutive grid points.
gn : unsigned int
    Number of grid points.

Returns
-------
bspline_val : float
    The B-spline value at x.
*/

WITHIN_KERNEL ${float}
bspline_value(const ${float} x,
              const ${float} gmin,
              const ${float} gstep,
              const unsigned int gn
              ){
    // Compute coordinates of x in the grid
    const ${float} x_idx_f = (x - gmin) / gstep;
    const int x_idx = (int) ${floor}(x_idx_f);
    const ${float} lam = x_idx_f - x_idx;

    // Determine weights
    ${float} a0, a1, a2, a3;
    bspline_weights(lam, &a0, &a1, &a2, &a3);

    // TO BE CONTINUED
}

/*************************************************************************/

KERNEL void
test_kernel(GLOBAL_MEM ${float} *out,
            const ${float} *x,
            const unsigned int n){
    unsigned int i;
    for (i=0; i<n; i++)
        out[i] = bspline3_norm_coord(x[i]);

    return;
}
""")


# --- Kernel test --- #

x = pygpu.gpuarray.array([0, 0.5, 1.0, 1.5, 2.0, 2.5])
src = kernel_tpl.render(float=DTYPE_TO_CTYPE[x.dtype],
                        floor='floor')
out = x._empty_like_me()
args = [ArrayArg(out.dtype, 'out'), ArrayArg(x.dtype, 'x'),
        ScalarArg(np.dtype('uint32'), 'n')]
spec = [pygpu.gpuarray.GpuArray, pygpu.gpuarray.GpuArray, 'uint32']
have_small = False
have_double = False
have_complex = False
for arg in args:
    if arg.dtype.itemsize < 4 and type(arg) == ArrayArg:
        have_small = True
    if arg.dtype in [np.float64, np.complex128]:
        have_double = True
    if arg.dtype in [np.complex64, np.complex128]:
        have_complex = True

flags = dict(have_small=have_small, have_double=have_double,
             have_complex=have_complex)

kernel = pygpu.gpuarray.GpuKernel(
    src, 'test_kernel', spec, context=out.context, cluda=True, **flags)

kernel(out, x, x.size, n=x.size)
