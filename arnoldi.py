#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 19:30:25 2017

@author: hkohr
"""

import odl
import numpy as np
import scipy.misc


def arnoldi(A, b, x0, n):
    assert x0 in A.domain
    assert A.domain == A.range
    h = np.zeros((n + 1, n))

    r = A(x0) - b
    q = [r / r.norm()]

    for i in range(n):
        v = A(q[i])
        for j in range(i):
            h[j, i] = q[j].inner(v)
            v -= h[j, i] * q[j]

        h[i + 1, i] = v.norm()
        q.append(v / h[i + 1, i])

    return h, q


class KrylovSpaceEmbedding(odl.Operator):

    def __init__(self, q):
        assert all(qi.space == q[0].space for qi in q[1:])
        super(KrylovSpaceEmbedding, self).__init__(
            domain=odl.rn(len(q)), range=q[0].space, linear=True)

        self.q = q

    def _call(self, z, out):
        out.lincomb(z[0], self.q[0])
        for i in range(1, len(z)):
            out.lincomb(1, out, z[i], self.q[i])
        return out

    @property
    def adjoint(self):
        op = self

        class KrylovSpaceEmbeddingAdjoint(odl.Operator):

            def __init__(self):
                super(KrylovSpaceEmbeddingAdjoint, self).__init__(
                    domain=op.range, range=op.domain, linear=True)

            def _call(self, x, out):
                for i in range(len(out)):
                    out[i] = op.q[i].inner(x)
                out *= op.range.weighting.const
                return out

            @property
            def adjoint(self):
                return op

        return KrylovSpaceEmbeddingAdjoint()


def admm_linearized_krylov(alpha, g, L, A, b, niter_arn, sigma, niter,
                           **kwargs):
    """Linearized ADMM using a Krylov subspace reformulation.

    Parameters
    ----------
    alpha : `FnBase` element
        Starting point of the iteration, updated in-place. It must be
        an element of an `rn` type space with ``n`` elements, where
        ``n`` is the Krylov subspace dimension.
    g : `Functional`
        The ``g`` function in the problem formulation. Needs to define
        ``g.proximal``.
    L : linear `Operator`
        Operator ``L`` in the problem formulation. It must fulfill
        ``L.range == g.domain``, and ``L.range`` must be either equal to
        or a product space of ``L.domain``.
    q : sequence of ``L.domain`` elements
        The vectors spanning the Krylov subspace in which the problem
        should be solved.
    h : array-like
        Upper Hessenberg matrix that is a result of the Arnoldi iteration
        to find the Krylov subspace. Its shape must be ``(n+1, n)``,
        where ``n = len(q) - 1`` is the dimension of the Krylov space.
    x0 : ``L.domain`` element
        Start estimate for the Arnoldi iteration.
    r : ``L.domain`` element
        Initial residual in the Arnoldi iteration.
    sigma : positive float
        Step size parameter for ADMM.
    niter : nonnegative int
        Number of iterations.

    Other Parameters
    ----------------
    kwargs :
        Further keyword arguments passed on to `admm_linearized`.

    Notes
    -----
    This method solves a problem of the form

    .. math::
        \min_x g(Lx)

    with a convex function :math:`g` and a linear operator
    :math:`L`.

    It uses a Krylov subspace to reduce the problem size as given by
    an upper Hessenberg matrix
    :math:`H_{n} \\in \mathbb{R}^{(n+1) \\times n}` and :math:`(n+1)`
    elements :math:`q_i \\in X`, where :math:`X` is the domain of
    :math:`L`. By defining a unitary operator

    .. math::
        Q_n: \mathbb{R}^n \\to X, \quad Q_n(\\alpha) =
            \sum_{i=1}^n \\alpha_i\, q_i

    the original problem is reformulated as

    .. math::
        \min_\\alpha g(L Q_{n+1} H_n \\alpha + L x_0)
        \quad \Leftrightarrow \quad
        \min_\\alpha \\tilde{g}(U \\alpha)

    with :math:`U = L Q_{n+1} H_n: \mathbb{R}^n \\to X` and
    :math:\\tilde{g}(y) = g(y + L x_0)`. Here, :math:`x_0` is the starting
    point of the Arnoldi iteration to compute the Krylov subspace.
    """
    h, q = arnoldi(A, b, x0, niter_arn)
    beta = (A(x0) - b).norm()

    Qn = KrylovSpaceEmbedding(q[:-1])
    Qnp1 = KrylovSpaceEmbedding(q)
    H = odl.MatrixOperator(h)
    assert alpha in H.domain
    assert Qnp1.domain == H.range
    assert L.domain == Qnp1.range

    g_transl = g.translated(-L(x0))

    U = L * Qn
    S = odl.BroadcastOperator(H, U)

    f = odl.solvers.ZeroFunctional(alpha.space)
    e1 = H.range.zero()
    e1[0] = 1
    data_fit = odl.solvers.L2NormSquared(H.range).translated(beta * e1)
    G = odl.solvers.SeparableSum(data_fit, g_transl)

    opnorm_H = odl.power_method_opnorm(H, maxiter=50)
    tau = 0.5 * sigma / opnorm_H ** 2
    odl.solvers.admm_linearized(alpha, f, G, S, tau, sigma, niter,
                                **kwargs)


# %%
image = scipy.misc.ascent()[::2, ::2].astype(float)
image /= image.max()
space = odl.uniform_discr([0, 0], image.T.shape, image.T.shape)
x = space.element(np.rot90(image, -1))

fourier = odl.trafos.FourierTransform(space)


def kernel_ft_func(x, **kwargs):
    s = kwargs.pop('s', 1.0)
    return (np.exp(-sum(s ** 2 * xi ** 2 / 2 for xi in x)) /
            (2 * np.pi) ** len(x))


kernel_ft = fourier.range.element(kernel_ft_func, s=2)
conv = fourier.inverse * kernel_ft * fourier

A = conv
b = A(x)
x0 = space.zero()

h, q = arnoldi(A, b, x0, 10)

grad = odl.Gradient(space)
g = odl.solvers.L1Norm(grad.range)
L = grad
