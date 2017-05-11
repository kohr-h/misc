import matplotlib.pyplot as plt
import numpy as np
import pickle
import odl
import os


# --- Load data and define ray transform --- #

# Define reco space
vol_size = np.array([230.0, 230.0])
vol_min = np.array([-115.0, -115.0])
shape = (512, 512)
reco_space = odl.uniform_discr(vol_min, vol_min + vol_size, shape)

# Set paths and file names
# data_path = '/home/hkohr/SciData/Head_CT_Sim/'
data_path = '/export/scratch2/kohr/data/Head_CT_Sim'
geom_fname = 'HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_2D.geometry.p'
data_fname = 'HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_2D_120kV.npy'

# Get geometry and data
with open(os.path.join(data_path, geom_fname), 'rb') as f:
    geom = pickle.load(f, encoding='latin1')

data_arr = np.load(os.path.join(data_path, data_fname)).astype('float32')
log_data_arr = -np.log(data_arr / np.max(data_arr))

# Define ray transform
ray_trafo = odl.tomo.RayTransform(reco_space, geom)

# Initialize data as ODL space element and display it, clipping to a
# somewhat reasonable range
# We need to rescale a bit to let the difference between squaring and
# not squaring not too big
scaling = 30.0
data = ray_trafo.range.element(
    log_data_arr / np.max(log_data_arr) * vol_size[0] * scaling)


# --- Perform FBP reconstruction --- #


#  Compute FBP reco for a good initial guess
fbp = odl.tomo.fbp_op(ray_trafo, padding=True, filter_type='Hamming',
                      frequency_scaling=0.6)
fbp_reco = fbp(data)


# --- Compute the exponent --- #


# We use the following procedure to generate the exponent from the reco g:
# - Compute a moderately smoothed version of the Laplacian L(g)
# - Take its absolute value and smooth it more aggressively
# - Multiply by 2 / max(L(g)), then clip at value 1.
#   This is to make the regions with high values broader.
# - Use 2 minus the result as exponent
def exp_kernel(x, **kwargs):
    s = kwargs.pop('s', 0.5)
    scaled = [xi / (np.sqrt(2) * s) for xi in x]
    return np.exp(-sum(xi ** 2 for xi in scaled))


# Pre-smoothing convolution
fourier = odl.trafos.FourierTransform(reco_space)
pre_kernel = reco_space.element(exp_kernel, s=1.5)
pre_kernel_ft = fourier(pre_kernel) * (2 * np.pi)
pre_conv = fourier.inverse * pre_kernel_ft * fourier
smoothed_lapl = odl.Laplacian(reco_space, pad_mode='symmetric') * pre_conv
# Smoothed Laplacian of the data
abs_lapl = np.abs(smoothed_lapl(fbp_reco))
# Remove jumps at the boundary, they're artificial
abs_lapl[:8, :] = 0
abs_lapl[-8:, :] = 0
abs_lapl[:, :8] = 0
abs_lapl[:, -8:] = 0
# Post-smoothing
post_kernel = reco_space.element(exp_kernel, s=1.5)
post_kernel_ft = fourier(post_kernel) * (2 * np.pi)
post_conv = fourier.inverse * post_kernel_ft * fourier
conv_abs_lapl = np.maximum(post_conv(abs_lapl), 0)
conv_abs_lapl -= np.min(conv_abs_lapl)
conv_abs_lapl *= 80 / np.max(conv_abs_lapl)
conv_abs_lapl[:] = np.minimum(conv_abs_lapl, 1)
# Cut off corners since they produce bad values
conv_abs_lapl[:80, :80] = 0
conv_abs_lapl[:80, -80:] = 0
conv_abs_lapl[-80:, :80] = 0
conv_abs_lapl[-80:, -80:] = 0
exponent = 2.0 - conv_abs_lapl
exponent.show()


# --- Set up the TV-regularized problem and solve it --- #

# Assemble operators and functionals for the solver
gradient = odl.Gradient(reco_space, pad_mode='order1')
lin_ops = [ray_trafo, gradient]
data_matching = odl.solvers.L2NormSquared(ray_trafo.range).translated(data)
varlp_func = odl.solvers.VariableLpModular(gradient.range, exponent,
                                           impl='gpuarray')

# Left-multiplication version
reg_param = 2e-2
# regularizer = reg_param * varlp_func
# Right-multiplication version
reg_param = 5e-2 * scaling
regularizer = varlp_func * reg_param

g = [data_matching, regularizer]
f = odl.solvers.IndicatorBox(reco_space, 0, 2.5 * scaling)

clim = np.array([1.05, 1.15]) * scaling
# Uncomment the combined callback to also display iterates
callback = (odl.solvers.CallbackShow('iterate', step=5, clim=clim) &
            odl.solvers.CallbackPrintIteration())
# callback = odl.solvers.CallbackPrintIteration()

# Solve with initial guess x = data.
# Step size parameters are selected to ensure convergence.
# See douglas_rachford_pd doc for more information.
x = fbp_reco.copy()


def check_douglas_rachford_pd_params(tau, sigma, lam, linop_norms):
    """Check if the selected parameters fulfill the convergence criterion."""
    lhs = tau * sum(sigma[i] * linop_norms[i] ** 2
                    for i in range(max(len(sigma), len(linop_norms))))

    if lhs >= 4:
        print('LHS of the criterion must be < 4.0, but evaluates to {}'
              ''.format(lhs))
        assert False
    else:
        print('LHS of the criterion must be < 4.0, evaluates to {}'
              ''.format(lhs))


lam = 1.5
ray_trafo_norm = odl.power_method_opnorm(ray_trafo, maxiter=4)
grad_norm = odl.power_method_opnorm(gradient, xstart=fbp_reco, maxiter=4)
norms = [ray_trafo_norm * 1.1, grad_norm * 1.1]
sigma = [0.002, 0.6]
tau = 0.1

check_douglas_rachford_pd_params(tau, sigma, lam, norms)
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops, tau, sigma, niter=300,
                                lam=lam, callback=callback)


# --- Display results --- #


data.show('Sinogram', clim=[0, 250 * scaling])
fbp_reco.show('FBP reconstruction', clim=clim)
x.show(title='Variable Lp TV Reconstruction', clim=clim)

# Create horizontal profile through the "tip"
fbp_slice = fbp_reco[:, 250]
reco_slice = x[:, 250]
x_vals = reco_space.grid.coord_vectors[0]
fig, axes = plt.subplots()
axes.set_ylim(clim)
plt.plot(x_vals, fbp_slice, label='FBP reconstruction')
axes.plot(x_vals, reco_slice, label='Variable Lp TV reconstruction')
plt.legend()
plt.tight_layout()
plt.savefig('head_phantom_varlp_tv_profile.png')


# Display full image
fig, axes = plt.subplots()
axes.imshow(np.rot90(x), cmap='bone', clim=clim)
axes.axis('off')
plt.tight_layout()
plt.savefig('head_phantom_varlp_tv_reco.png')
