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
data = ray_trafo.range.element(log_data_arr)


# --- Perform FBP reconstruction --- #


#  Compute FBP reco for a good initial guess
fbp = odl.tomo.fbp_op(ray_trafo, padding=True, filter_type='Hamming',
                      frequency_scaling=0.8)
fbp_reco = fbp(data)


# --- Set up the TV-regularized problem and solve it --- #


# Assemble operators and functionals for the solver
gradient = odl.Gradient(reco_space, pad_mode='order1')
lin_ops = [ray_trafo, gradient]
data_matching = odl.solvers.L2NormSquared(ray_trafo.range).translated(data)
l1_func = odl.solvers.L1Norm(gradient.range)

reg_param = 2e-3
regularizer = reg_param * l1_func

g = [data_matching, regularizer]
f = odl.solvers.IndicatorBox(reco_space, 0, 0.04)

# Uncomment the combined callback to also display iterates
callback = (odl.solvers.CallbackShow('iterate', step=5, clim=[0.019, 0.022]) &
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


tau = 0.2
sigma = [0.002, 0.4]
lam = 1.5
ray_trafo_norm = odl.power_method_opnorm(ray_trafo, maxiter=4)
grad_norm = odl.power_method_opnorm(gradient, xstart=fbp_reco, maxiter=4)
norms = [ray_trafo_norm * 1.1, grad_norm * 1.1]

check_douglas_rachford_pd_params(tau, sigma, lam, norms)
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops, tau, sigma, niter=300,
                                lam=lam, callback=callback)


# --- Display results --- #


data.show('Sinogram', clim=[0, 4.5])
fbp_reco.show('FBP reconstruction', clim=[0.019, 0.022])
x.show(title='TV Reconstruction', clim=[0.019, 0.022])

# Create horizontal profile through the "tip"
fbp_slice = fbp_reco[:, 250]
reco_slice = x[:, 250]
x_vals = reco_space.grid.coord_vectors[0]
fig, axes = plt.subplots()
axes.set_ylim([0.019, 0.029])
plt.plot(x_vals, fbp_slice, label='FBP reconstruction')
axes.plot(x_vals, reco_slice, label='TV reconstruction')
plt.legend()
plt.tight_layout()
plt.savefig('head_phantom_tv_profile.png')


# Display full image
fig, axes = plt.subplots()
axes.imshow(np.rot90(x), cmap='bone', clim=[0.019, 0.022])
axes.axis('off')
plt.tight_layout()
plt.savefig('head_phantom_tv_reco.png')
