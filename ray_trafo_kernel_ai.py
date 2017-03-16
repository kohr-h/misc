"""Example using the ray transform with 2d parallel beam geometry."""

import numpy as np
import odl

# Discrete reconstruction space: discretized functions on the cube
# [-20, 20]^2 x [0, 40] with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20, 0], max_pt=[20, 20, 40], shape=[300, 300, 300],
    dtype='float32')

# Make a helical cone beam geometry with flat detector
# Angles: uniformly spaced, n = 2000, min = 0, max = 8 * 2 * pi
angle_partition = odl.uniform_partition(0, 8 * 2 * np.pi, 2000)
# Detector: uniformly sampled, n = (558, 60), min = (-50, -3), max = (50, 3)
detector_partition = odl.uniform_partition([-50, -3], [50, 3], [558, 60])
# Spiral has a pitch of 5, we run 8 rounds (due to max angle = 8 * 2 * pi)
geometry = odl.tomo.HelicalConeFlatGeometry(
    angle_partition, detector_partition, src_radius=100, det_radius=100,
    pitch=5.0)

# Ray transform (= forward projection).
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)


gamma = 0.3


def gaussian(x):
    scaled_squares = [-xi ** 2 / (2 * gamma ** 2) for xi in x]
    return np.exp(sum(scaled_squares)) / (np.sqrt(2 * p) * gamma) ** len(x)


mollifier = reco_space.element(gaussian)
it = 0


def print_residual(x):
    global it
    residual = ray_trafo(ray_trafo.adjoint(x) - mollifier)
    print('Iteration {}: residual {}'.format(it, residual.norm()))
    it += 1


callback = (odl.solvers.CallbackShow(step=10) &
            odl.solvers.CallbackApply(print_residual, step=10))


# Solve for the reconstruction kernel
reco_ker = ray_trafo.adjoint.domain.zero()
odl.solvers.conjugate_gradient_normal(ray_trafo.adjoint, reco_ker,
                                      mollifier, niter=50, callback=callback)


fourier = odl.trafos.FourierTransform(ray_trafo.range, axes=[1, 2],
                                      impl='pyfftw', halfcomplex=False)
reco_ker_ft = fourier(reco_ker)
reco_ker_conv = fourier.inverse * reco_ker_ft * fourier * (2 * np.pi) ** 1.5

fbp = ray_trafo.adjoint * reco_ker_conv

phantom = odl.phantom.shepp_logan(reco_space, modified=True)
data = ray_trafo(phantom)
reco = fbp(data)

mollifier.show('Mollifier')
reco_ker.show('Reconstruction kernel')
data.show('Data')
reco.show('Reconstruction')
