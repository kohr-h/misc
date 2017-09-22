import numpy as np
import odl


def binary_reco(x, lam, fwd_op, data, smooth_op, mus, niter_inner,
                callback_primal=None, callback_dual=None,
                callback_loop='inner'):

    # Set subgradient `x \in subgrad[g^*](y)`
    def gstar_subgrad(y, out):
        out.assign(y)

        with odl.util.writable_array(out) as x:
            x[x <= 0] = 0
            x[x >= lam] = 1
            x[(x > 0) & (x < lam)] /= lam

        return out

    def h_grad(x, mu, out):
        # TODO: optimize
        data_term = fwd_op.adjoint(fwd_op(x) - data)
        smooth_term = smooth_op.adjoint(smooth_op(x))
        out[:] = (lam + mu) * x - data_term - smooth_term - mu / 2

    y = x.space.element()

    for mu in mus:
        for it_inner in range(niter_inner):
            h_grad(x, mu, out=y)
            gstar_subgrad(y, out=x)

            if callback_loop == 'inner':
                if callback_primal is not None:
                    callback_primal(x)
                if callback_dual is not None:
                    callback_dual(y)

        if callback_loop == 'outer':
            if callback_primal is not None:
                callback_primal(x)
            if callback_dual is not None:
                callback_dual(y)


# %%

# --- Define setting --- #

np.random.seed(123)

space = odl.uniform_discr([-1, -1], [1, 1], (100, 100))


# Make a binary phantom
def bin_func(x):
    return (((x[0] >= 0) & (x[0] <= 0.5) & (x[1] >= -0.5) & (x[1] <= 0.5)) |
            ((x[0] + 0.5) ** 2 + (x[1] - 0.25) ** 2 <= 0.2 ** 2) |
            ((x[0] + 0.5) ** 2 + (x[1] + 0.25) ** 2 <= 0.2 ** 2))


phantom = space.element(bin_func)

# Make noisy data
data = phantom + odl.phantom.white_noise(space, stddev=0.2)

# Forward operator
fwd_op = odl.IdentityOperator(space)

# Smoothness operator with regularization parameter
smooth_op = odl.Gradient(space, pad_mode='order1')
alpha = space.cell_sides[0] ** 2 / 5

# Compute operator norms to determine a suitable lambda
fwd_op_norm = odl.power_method_opnorm(fwd_op, maxiter=2)
smooth_op_norm = odl.power_method_opnorm(smooth_op, xstart=data, maxiter=20)
lam = 4 * (fwd_op_norm ** 2 + alpha * smooth_op_norm ** 2)

# Define mu parameters
mus = np.linspace(0, 1, 20)

# Callbacks to see something
callback_primal = odl.solvers.CallbackShow('Primal')
callback_dual = odl.solvers.CallbackShow('Dual')

# Call into the reco method
x = space.zero()
binary_reco(x, lam, fwd_op, data, alpha * smooth_op, mus, niter_inner=20,
            callback_primal=callback_primal, callback_dual=callback_dual,
            callback_loop='inner')
