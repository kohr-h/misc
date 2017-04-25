#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:43:58 2017

@author: kohr
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy


def spline_tangents_natural(points):
    # Points stacked along axis 0, at least 2
    points = np.array(points, copy=False, ndmin=2)
    assert points.ndim == 2
    assert points.shape[0] >= 2
    ndim = points.shape[1]

    # matrix for spline coefficients
    n_points = points.shape[0]
    matrix = np.zeros((n_points, n_points))
    # Set inner matrix block
    idcs = np.arange(n_points)
    matrix[idcs[1:-1], idcs[1:-1]] = 4  # diag, inner part
    matrix[idcs[1:-1], idcs[:-2]] = 1  # lower diag
    matrix[idcs[1:-1], idcs[2:]] = 1  # upper diag
    # Set first and last rows
    matrix[0, 0] = 2
    matrix[0, 1] = 1
    matrix[-1, -1] = 2
    matrix[-1, -2] = 1

    # Generate RHS
    rhs = np.empty_like(points)
    rhs[1:-1] = 3 * (points[2:] - points[:-2])
    rhs[0] = 3 * (points[1] - points[0])
    rhs[-1] = 3 * (points[-1] - points[-2])

    # Solve for the tangent vectors
    tangents = scipy.linalg.solve(matrix, rhs)
    assert tangents.shape == (n_points, ndim)
    return tangents


def spline_tangents_clamp(points, dx_left=None, dx_right=None):
    # Points stacked along axis 0, at least 2
    points = np.array(points, copy=False, ndmin=2)
    assert points.ndim == 2
    assert points.shape[0] >= 2
    ndim = points.shape[1]

    # Left and right derivatives should be single points in `ndim` dimensions
    if dx_left is None:
        dx_left = points[1] - points[0]
    else:
        dx_left = np.asarray(dx_left)
    if dx_right is None:
        dx_right = points[-1] - points[-2]
    else:
        dx_right = np.asarray(dx_right)

    assert dx_left.shape == (ndim,)
    assert dx_right.shape == (ndim,)

    # matrix for spline coefficients
    n_points = points.shape[0]
    matrix = np.zeros((n_points, n_points))
    # Set inner matrix block
    idcs = np.arange(n_points)
    matrix[idcs[1:-1], idcs[1:-1]] = 4  # diag, inner part
    matrix[idcs[1:-1], idcs[:-2]] = 1  # lower diag
    matrix[idcs[1:-1], idcs[2:]] = 1  # upper diag
    # Set first and last rows
    matrix[0, 0] = 1
    matrix[-1, -1] = 1

    # Generate RHS
    rhs = np.empty_like(points)
    rhs[1:-1] = 3 * (points[2:] - points[:-2])
    rhs[0] = dx_left
    rhs[-1] = dx_right

    # Solve for the tangent vectors
    tangents = scipy.linalg.solve(matrix, rhs)
    assert tangents.shape == (n_points, ndim)
    return tangents


def spline_tangents_novert(points):
    # Points stacked along axis 0, at least 2
    points = np.array(points, copy=False, ndmin=2)
    assert points.ndim == 2
    assert points.shape[0] >= 2
    ndim = points.shape[1]

    # Left and right derivatives are set to the vectors between first
    # and last point pairs, except for the last axis, where the derivatives
    # are set to 0
    dx_left = points[1] - points[0]
    dx_left[-1] = 0
    dx_right = points[-1] - points[-2]
    dx_right[-1] = 0

    # matrix for spline coefficients
    n_points = points.shape[0]
    matrix = np.zeros((n_points, n_points))
    # Set inner matrix block
    idcs = np.arange(n_points)
    matrix[idcs[1:-1], idcs[1:-1]] = 4  # diag, inner part
    matrix[idcs[1:-1], idcs[:-2]] = 1  # lower diag
    matrix[idcs[1:-1], idcs[2:]] = 1  # upper diag
    # Set first and last rows
    matrix[0, 0] = 1
    matrix[-1, -1] = 1

    # Generate RHS
    rhs = np.empty_like(points)
    rhs[1:-1] = 3 * (points[2:] - points[:-2])
    rhs[0] = dx_left
    rhs[-1] = dx_right

    # Solve for the tangent vectors
    tangents = scipy.linalg.solve(matrix, rhs)
    assert tangents.shape == (n_points, ndim)
    return tangents


def spline(s, points, tangents):
    # Parameter should be a single value or a 1d array
    s = np.array(s, copy=False, ndmin=1)
    n_params = s.shape[0]
    assert s.ndim == 1

    # points are (n_splines + 1, ndim)-shaped
    points = np.array(points, copy=False, ndmin=2)
    assert points.ndim == 2
    n_splines = points.shape[0] - 1
    ndim = points.shape[1]

    # tangents have the same shape as points
    tangents = np.array(tangents, copy=False, ndmin=2)
    assert tangents.shape == points.shape

    # Make sure the parameter values are in [0, n_splines]
    assert np.all((s >= 0) & (s <= n_splines))

    # Generate spline points, (n_params, ndim)-shaped
    spline_points = np.empty((n_params, ndim))
    point_diffs = np.diff(points, axis=0)
    n_assigned = 0
    for i in range(n_splines):
        # Normalize parameter to [0, 1), for the last spline to [0, 1]
        if i < n_splines - 1:
            cond = (s >= i) & (s < i + 1)
        else:
            cond = (s >= i) & (s <= i + 1)
        norm_s = s[cond] - i

        # Compute help vectors for spline points
        n_params_i = len(norm_s)
        a = tangents[i] - point_diffs[i]
        b = point_diffs[i] - tangents[i + 1]

        # Broadcasting with s along first axis and ndim along second axis
        norm_s = norm_s[:, None]
        pt = points[i][None, :]
        pt_diff = point_diffs[i][None, :]
        a = a[None, :]
        b = b[None, :]

        # Compute spline points as
        # p(s) = x[k] + s(x[k+1] - x[k]) + s(1-s)((1-s)a + sb)
        # See https://en.wikipedia.org/wiki/Spline_interpolation
        spl_pts = (pt + norm_s * pt_diff +
                   norm_s * (1 - norm_s) * ((1 - norm_s) * a + norm_s * b))

        # Assing to spline_points and increment n_assigned
        spline_points[n_assigned:n_assigned + n_params_i] = spl_pts
        n_assigned += n_params_i

    return spline_points


def arc_length(spline_pts):
    spline_pts = np.array(spline_pts, copy=False, ndmin=2)
    assert spline_pts.ndim == 2

    diffs = np.diff(spline_pts, axis=0)
    diff_norms = np.linalg.norm(diffs, axis=1)

    lengths = np.empty(spline_pts.shape[0])
    lengths[0] = 0
    lengths[1:] = np.cumsum(diff_norms)
    return lengths


# %% Test code

points = np.array([[0, -1], [-1, 0], [1, 1], [-2, 1], [-4, 0]])
n_splines = len(points) - 1
n_eval_pts = n_splines * 50
tangents_natural = spline_tangents_natural(points)
tangents_clamp = spline_tangents_clamp(points)
tangents_novert = spline_tangents_novert(points)
s = np.linspace(0, n_splines, num=n_eval_pts)

spline_pts_natural = spline(s, points, tangents_natural)
spline_pts_clamp = spline(s, points, tangents_clamp)
spline_pts_novert = spline(s, points, tangents_novert)

# Plot stuff
fig, ax = plt.subplots()
ax.scatter(spline_pts_natural[:, 0], spline_pts_natural[:, 1], s=5)
ax.scatter(spline_pts_clamp[:, 0], spline_pts_clamp[:, 1], s=5)
ax.scatter(spline_pts_novert[:, 0], spline_pts_novert[:, 1], s=5)
ax.scatter(points[:, 0], points[:, 1], s=50, marker='s')
plt.title('Spline curves through given points')
ax.legend(['natural spline', 'clamp spline', 'novert spline', 'points'])

# Compute arc lengths and plot them as functions of s
arc_len_natural = arc_length(spline_pts_natural)
arc_len_clamp = arc_length(spline_pts_clamp)
arc_len_novert = arc_length(spline_pts_novert)

fig, ax = plt.subplots()
ax.plot(s, arc_len_natural)
ax.plot(s, arc_len_clamp)
ax.plot(s, arc_len_novert)
plt.title('Arc length functions')
ax.legend(['natural spline', 'clamp spline', 'novert spline'])

# Set equispaced target length parameters between 0 and total arc length
total_arc_len = arc_len_natural[-1]
len_params = np.linspace(0, total_arc_len, num=n_eval_pts)  # l
len_idcs = np.searchsorted(arc_len_natural, len_params, side='right')  # alpha
len_idcs[len_idcs == n_eval_pts] = n_eval_pts - 1
weights = ((len_params - arc_len_natural[len_idcs - 1]) /
           (arc_len_natural[len_idcs] - arc_len_natural[len_idcs - 1]))  # nu
invfun_params = ((1 - weights) * s[len_idcs - 1] + weights * s[len_idcs])

fig, ax = plt.subplots()
ax.scatter(invfun_spline_pts_natural[:, 0], invfun_spline_pts_natural[:, 1],
           s=5)
ax.scatter(invfun_pts_natural[:, 0], invfun_pts_natural[:, 1], s=5)
ax.legend(['arc length spline', 'arc length points'])
