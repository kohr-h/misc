# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Variable Lp norm."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np
import odl
from odl.solvers.functional.default_functionals import VariableLpModular


class VariableLpNorm(odl.solvers.Functional):

    """The p-norm with spatially varying exponent ``p``.

    The variable Lp norm is defined as

        ``||f||_p = inf{s > 0 | rho_p(f / s) <= 1}``

    where ``rho_p`` is the variable Lp modular. Starting from the
    initial guess ``s = rho_p(f)``, a bisection method is used to
    determine the optimal ``s``.
    """

    def __init__(self, space, var_exp):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp`
            Discretized function space on which the modular is defined
        var_exp : scalar-valued ``space`` `element-like`
            The variable exponent ``p(x)``
        """
        super().__init__(space, linear=False)
        self.var_exp = self.domain.element(var_exp)
        self._min_exp = np.min(self.var_exp)
        self.modular = VariableLpModular(space, var_exp)

    def _call(self, f, **kwargs):
        """Return ``self(f)``.

        Parameters
        ----------
        f : `DiscreteLpVector`
            Function whose norm to calculate
        atol : positive `float`, optional
            Stop the iteration in the norm computation when
            ``|rho_p(f / s) - 1| <= atol``.
            Default: 0.01
        maxiter : positive `int`, optional
            Iterate at most this many times. Default: 10
        """
        atol = kwargs.pop('atol', 0.01)
        maxiter = kwargs.pop('maxiter', 10)

        s = self.modular(f)
        if s == 0:
            return 0.0

        m = self.modular(f / s)
        if abs(m - 1) <= atol:
            return s
        elif m < 1:
            fac = 0.5
        else:
            fac = 2.0

        # Find a starting point for the s iteration
        m_old = m
        s_old = s
        it = 0
        while True:
            s *= fac
            m = self.modular(f / s)
            it += 1
            if np.sign(m - 1) != np.sign(m_old - 1):
                break
            else:
                m_old = m
                s_old = s

        # Iterate until tolerance or maximum number of iterations reached
        s_low, s_up = min(s, s_old), max(s, s_old)
        for _ in range(maxiter - it + 1):
            s_test = (s_low + s_up) / 2  # TODO: use golden ratio
            m_test = self.modular(f / s_test)
            if abs(m_test - 1) <= atol:
                return s_test
            elif m_test < 1:
                s_up = s_test
            else:
                s_low = s_test

        return (s_low + s_up) / 2


class VariableLpUnitBallProjector(odl.Operator):

    """Projector onto the unit ball ``{f: ||f||_p <= 1}``.

    Currently, we implement the simplified version

        ``P(f) = f / ||f||_p``.
    """

    def __init__(self, norm_func):
        """Initialize a new instance.

        Parameters
        ----------
        norm_func : `VariableLpNorm`
            Functional to evaluate the norm
        """
        self.norm = norm_func
        super().__init__(self.norm.domain, self.norm.domain, linear=False)

    def _call(self, x):
        """Return ``self(x)``."""
        # TODO: this is a bogus implementation, replace it!
        return x / self.norm(x)
