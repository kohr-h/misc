import numpy as np
import odl


# --- Define operator and center of gravity functional --- #

class Abs2Operator(odl.Operator):
    """Operator implementing ``f -> f.real ** 2 + f.imag ** 2``."""

    def __init__(self, space):
        # Operator mapping from complex to real
        super(Abs2Operator, self).__init__(space, space.real_space)

    def _call(self, f):
        # f is complex, output is real
        return f.real ** 2 + f.imag ** 2

    def derivative(self, f):
        op = self

        class Abs2OperatorDerivative(odl.Operator):
            def __init__(self):
                super(Abs2OperatorDerivative, self).__init__(
                    op.domain, op.range, linear=True)

            def _call(self, x):
                # f and x are both complex, output is real
                return 2 * (f.real * x.real + f.imag * x.imag)

            def norm(self):
                return 2 * f.norm()

            @property
            def adjoint(self):
                deriv = self

                class Abs2OperatorDerivativeAdjoint(odl.Operator):
                    def __init__(self):
                        super(Abs2OperatorDerivativeAdjoint, self).__init__(
                            deriv.range, deriv.domain, linear=True)

                    def _call(self, u, out):
                        out.real = 2 * f.real * u
                        out.imag = 2 * f.imag * u

                    @property
                    def adjoint(self):
                        return deriv

                return Abs2OperatorDerivativeAdjoint()

        return Abs2OperatorDerivative()


class CenterOfGravity(odl.Operator):

    def __init__(self, space):
        ran_type = odl.rn if space.is_rn else odl.cn
        super(CenterOfGravity, self).__init__(
            domain=space, range=ran_type(space.ndim), linear=True)

    def _call(self, f, out):
        x = self.domain.meshgrid
        for i, xi in enumerate(x):
            fac = self.domain.element(np.broadcast_to(xi, space.shape))
            out[i] = f.inner(fac)

    @property
    def adjoint(self):
        op = self

        class CenterOfGravityAdjoint(odl.Operator):
            def __init__(self):
                super(CenterOfGravityAdjoint, self).__init__(
                    op.range, op.domain, linear=True)

            def _call(self, y):
                x = self.range.meshgrid
                return (sum(xi * yi for xi, yi in zip(x, y)) /
                        op.domain.weighting.const)

            @property
            def adjoint(self):
                return op

        return CenterOfGravityAdjoint()


class IndicatorSupport(odl.solvers.Functional):

    def __init__(self, space, support):
        super(IndicatorSupport, self).__init__(space)
        self.support = support
        self.not_support = 1 - support

    def _call(self, f):
        masked = f.copy()
        if self.domain.is_rn:
            masked *= self.not_support
        else:
            masked.real *= self.not_support
            masked.imag *= self.not_support

        return np.inf if np.any(np.nonzero(masked)) else 0.0

    def proximal(self, sigma):
        if self.domain.is_rn:
            mult = self.support
        else:
            mult = self.domain.element()
            mult.real = self.support
            mult.imag = self.support

        return odl.MultiplyOperator(domain=self.domain, range=self.domain,
                                    multiplicand=mult)


class IndicatorSupportAndBox(odl.solvers.Functional):

    def __init__(self, space, support_mask, lower=None, upper=None):
        super(IndicatorSupportAndBox, self).__init__(space)
        self.ind_supp = IndicatorSupport(space, support_mask)
        self.ind_box = odl.solvers.IndicatorBox(space, lower, upper)

    def _call(self, f):
        return self.ind_supp(f) + self.ind_box(f)

    def proximal(self, sigma):
        return self.ind_box.proximal(sigma) * self.ind_supp.proximal(sigma)


# --- Set up space, forward operator and data --- #

space = odl.uniform_discr([-1, -1], [1, 1], (256, 256), dtype='complex64')
fourier = odl.trafos.FourierTransform(space)
abs2 = Abs2Operator(fourier.range)
far_field_op = abs2 * fourier

# Pure phase object
phantom = odl.phantom.shepp_logan(space, modified=True)
with odl.util.writable_array(phantom) as p_arr:
    p_arr[p_arr < 0] = 0
far_field = far_field_op(phantom)

# --- Set up the inverse problem --- #

# Gradient operator for the TV part
grad = odl.Gradient(space)

# Stacking of the two operators
L = odl.BroadcastOperator(far_field_op, grad)

# Data matching functional
data_fit = odl.solvers.L2NormSquared(far_field_op.range).translated(far_field)
# Regularization functional, the L1 norm
reg_func = 1e-5 * odl.solvers.GroupL1Norm(grad.range)
# Force support to be inside an ellipsoid
shepp_logan_ellipses = odl.phantom.transmission.shepp_logan_ellipsoids(
    ndim=2, modified=True)
supp_func = odl.phantom.ellipsoid_phantom(space.real_space,
                                          shepp_logan_ellipses[:1])
indicator = IndicatorSupportAndBox(space, supp_func, lower=0, upper=1 + 0.01j)
ind_supp = IndicatorSupport(space, supp_func)

f = indicator
g = odl.solvers.SeparableSum(data_fit, reg_func)

# --- Select parameters and solve using ADMM --- #

niter = 1000  # Number of iterations
delta = 0.1  # Step size for the constraint update

# Optionally pass a callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration(step=20) &
            odl.solvers.CallbackShow(step=20))

# Choose a starting point
#phaseless_data = far_field_op.domain.element(far_field)
#x = fourier.inverse(far_field)

#x = 0.5 * space.one()
#x.imag = 0
#for _ in range(niter // 10):
#    opnorm = odl.power_method_opnorm(far_field_op.derivative(x), xstart=x,
#                                     maxiter=2)
#    print('Operator norm:', opnorm)
#    odl.solvers.landweber(far_field_op, x, far_field, niter=10,
#                          omega=1.0 / opnorm ** 2,
#                          # projection=None,
#                          projection=lambda x: indicator.proximal(1)(x, out=x),
#                          callback=callback)

# Choose a starting point
x = 0.5 * space.one()
x.imag = 0

# Run the algorithm
odl.solvers.nonsmooth.admm.admm_precon_nonlinear(
    x, f, g, L, delta, niter, opnorm_factor=0.01, callback=callback)

# Display images
#phantom.show(title='Phantom')
#far_field.show('Far field data (range enhanced)', clim=[0, 0.0001])
#x.show(title='TV reconstruction', force_show=True)
