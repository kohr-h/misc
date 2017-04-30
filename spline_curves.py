#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:43:58 2017

@author: kohr
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy


class CubicSpline(object):
    def __init__(self, points, variant='natural', **kwargs):
        # points are (n_splines + 1, ndim)-shaped
        self.points = np.array(points, copy=False, ndmin=2)
        assert self.points.ndim == 2
        self.n_splines = points.shape[0] - 1
        assert self.n_splines > 0
        self.ndim = points.shape[1]
        self.variants = self._parse_variant(variant)

        # Tangents at the left and right ends
        tangent_left = kwargs.pop('tangent_left', None)
        if tangent_left is None:
            self.tangent_left = self.points[1] - self.points[0]
        else:
            self.tangent_left = np.asarray(tangent_left)
            assert self.tangent_left.shape == (self.ndim,)

        tangent_right = kwargs.pop('tangent_right', None)
        if tangent_right is None:
            self.tangent_right = self.points[-1] - self.points[-2]
        else:
            self.tangent_left = np.asarray(tangent_left)
            assert self.tangent_left.shape == (self.ndim,)

        # Tangent vectors at the points
        self.tangents = self._compute_tangents()

        # Some more useful quantities

        # Finite diffs between points
        self.point_diffs = np.diff(self.points, axis=0)

    def _parse_variant(self, variant):
        try:
            variant + ''
        except TypeError:
            assert len(variant) == self.ndim
            variants = []
            for var in variant:
                try:
                    var + ''
                except TypeError:
                    if len(var) == 1:
                        variants.append((str(var[0]), str(var[0])))
                    elif len(var) == 2:
                        variants.append((str(var[0]), str(var[1])))
                    else:
                        assert False
                else:
                    variants.append((str(var), str(var)))
        else:
            variants = [(str(variant), str(variant))] * self.ndim

        return tuple(variants)

    def _compute_tangents(self):
        # matrix for spline coefficients
        n_points = points.shape[0]
        matrix = np.zeros((n_points, n_points))
        # Set inner matrix block
        idcs = np.arange(n_points)
        matrix[idcs[1:-1], idcs[1:-1]] = 4  # diag, inner part
        matrix[idcs[1:-1], idcs[:-2]] = 1  # lower diag
        matrix[idcs[1:-1], idcs[2:]] = 1  # upper diag

        tangents = []
        rhs = np.empty(n_points)
        for i in range(self.ndim):
            # Set right-hand side, inner part
            rhs[1:-1] = 3 * (points[2:, i] - points[:-2, i])

            # Reset first and last matrix rows
            matrix[0] = 0
            matrix[-1] = 0
            # Set first equation according to left variant
            var_l, var_r = self.variants[i]
            if var_l == 'natural':
                matrix[0, 0] = 2
                matrix[0, 1] = 1
                rhs[0] = 3 * (points[1, i] - points[0, i])
            elif var_l == 'clamp':
                matrix[0, 0] = 1
                rhs[0] = self.tangent_left[i]
            elif var_l == 'zero':
                matrix[0, 0] = 1
                rhs[0] = 0
            else:
                raise ValueError('axis {}: bad left variant {!r}'
                                 ''.format(i, var_l))

            # Set last equation according to right variant
            if var_r == 'natural':
                matrix[-1, -1] = 2
                matrix[-1, -2] = 1
                rhs[-1] = 3 * (points[-1, i] - points[-2, i])
            elif var_r == 'clamp':
                matrix[-1, -1] = 1
                rhs[-1] = self.tangent_right[i]
            elif var_r == 'zero':
                matrix[-1, -1] = 1
                rhs[-1] = 0
            else:
                raise ValueError('axis {}: bad right variant {!r}'
                                 ''.format(i, var_r))

            tangents.append(scipy.linalg.solve(matrix, rhs))

        return np.vstack(tangents).T

    def arc_length(self, x):
        spline_pts = self(x)
        diffs = np.diff(spline_pts, axis=0)
        diff_norms = np.linalg.norm(diffs, axis=1)
        lengths = np.empty(spline_pts.shape[0])
        lengths[0] = 0
        lengths[1:] = np.cumsum(diff_norms)
        return lengths

    def __call__(self, x):
        # x should be a single value or a 1d array
        x = np.array(x, copy=False, ndmin=1)
        assert x.ndim == 1
        n_eval = x.shape[0]
        # Make sure the parameter values are in [0, n_splines]
        assert np.all((x >= 0) & (x <= self.n_splines))

        # Generate spline points, (n_eval, ndim)-shaped
        spline_points = np.empty((n_eval, self.ndim))
        n_assigned = 0
        for i in range(self.n_splines):
            # Normalize parameter to [0, 1), for the last spline to [0, 1]
            if i < n_splines - 1:
                cond = (x >= i) & (x < i + 1)
            else:
                cond = (x >= i) & (x <= i + 1)
            s = x[cond] - i

            # Compute help vectors for spline points
            n_eval_i = len(s)
            a = self.tangents[i] - self.point_diffs[i]
            b = self.point_diffs[i] - self.tangents[i + 1]

            # Broadcasting with x along first axis and ndim along second axis
            s = s[:, None]
            pt = self.points[i][None, :]
            pt_diff = self.point_diffs[i][None, :]
            a = a[None, :]
            b = b[None, :]

            # Compute spline points as
            # p(s) = p[k] + s(p[k+1] - p[k]) + s(1-s)((1-s)a + sb)
            # See https://en.wikipedia.org/wiki/Spline_interpolation
            spl_pts = pt + s * pt_diff + s * (1 - s) * ((1 - s) * a + s * b)

            # Assing to spline_points and increment n_assigned
            spline_points[n_assigned:n_assigned + n_eval_i] = spl_pts
            n_assigned += n_eval_i

        return spline_points

    def arc_length_params(self, x, alen):
        # Interpolate parameters x such that self.arg_length(x) = alen

        # x should be a single value or a 1d array
        x = np.array(x, copy=False, ndmin=1)
        assert x.ndim == 1
        n_eval = x.shape[0]

        # alen should be a single value or a 1d array
        alen = np.array(alen, copy=False, ndmin=1)
        assert alen.ndim == 1

        # Make sure the lengths are in [0, total_length]
        arc_lengths = self.arc_length(x)
        total_length = arc_lengths[-1]
        assert np.all((alen >= 0) & (alen <= total_length))

        idcs = np.searchsorted(arc_lengths, alen, side='right')  # alpha
        idcs[idcs == n_eval] = n_eval - 1
        weights = ((alen - arc_lengths[idcs - 1]) /
                   (arc_lengths[idcs] - arc_lengths[idcs - 1]))  # nu
        return ((1 - weights) * x[idcs - 1] + weights * x[idcs])  # r


# %% Input values for later

# Input values
points = np.array([[0, 0], [1, 1], [-1, 2], [0, 2], [5, 5]])
n_splines = len(points) - 1
n_eval_pts = n_splines * 50
x = np.linspace(0, n_splines, num=n_eval_pts)

# %% Test spline evaluation

# Create splines and evaluation points
spl_nat = CubicSpline(points, variant='natural')
spl_nat_pts = spl_nat(x)
spl_cl = CubicSpline(points, variant='clamp')
spl_cl_pts = spl_cl(x)
spl_nat_novert = CubicSpline(points, variant=['natural', 'zero'])
spl_nat_novert_pts = spl_nat_novert(x)
spl_cl_novert = CubicSpline(points, variant=['clamp', 'zero'])
spl_cl_novert_pts = spl_cl_novert(x)

# Plot stuff
fig, ax = plt.subplots()
ax.scatter(spl_nat_pts[:, 0], spl_nat_pts[:, 1], s=5)
ax.scatter(spl_cl_pts[:, 0], spl_cl_pts[:, 1], s=5)
ax.scatter(spl_nat_novert_pts[:, 0], spl_nat_novert_pts[:, 1], s=5)
ax.scatter(spl_cl_novert_pts[:, 0], spl_cl_novert_pts[:, 1], s=5)
ax.scatter(points[:, 0], points[:, 1], s=50, marker='s')
plt.title('Spline curves through given points')
ax.legend(['natural', 'clamp', 'natural, vertical 0', 'clamp, vertical 0',
           'points'])

# %% Arc length functionality

# Compute arc lengths and plot them as functions of s
arc_len_nat = spl_nat.arc_length(x)
arc_len_cl = spl_cl.arc_length(x)
arc_len_nat_novert = spl_nat_novert.arc_length(x)
arc_len_cl_novert = spl_cl_novert.arc_length(x)

fig, ax = plt.subplots()
ax.plot(x, arc_len_nat)
ax.plot(x, arc_len_cl)
ax.plot(x, arc_len_nat_novert)
ax.plot(x, arc_len_cl_novert)
plt.title('arc length functions')
ax.legend(['natural', 'clamp', 'natural, vertical 0', 'clamp, vertical 0'])

# Set equispaced target length parameters between 0 and total arc length
total_arc_len_nat = arc_len_nat[-1]
lengths_nat = np.linspace(0, total_arc_len_nat, num=n_eval_pts)
invfun_params = spl_nat.arc_length_params(x, lengths_nat)

# Check if the new parameters lead to samples that are equispaced with
# respect to arc length
check_spline_pts = spl_nat(invfun_params)
check_arc_len = spl_nat.arc_length(invfun_params)

fig, ax = plt.subplots()
ax.plot(check_arc_len)
ax.legend(['this should be a linear function'])
plt.title('arc length function after reparametrization')

# Check if the curves are still "the same"
fig, ax = plt.subplots()
ax.scatter(check_spline_pts[:, 0], check_spline_pts[:, 1], s=5)
ax.scatter(spl_nat_pts[:, 0], spl_nat_pts[:, 1], s=5)
ax.legend(['spline parametrized w/ arc length', 'original spline'])
