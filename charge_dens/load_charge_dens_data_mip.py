import numpy as np
import odl
import os

path = '/export/scratch2/kohr/data/Charge_Density/MIP'
fname = 'MIP_LH31'

with odl.tomo.FileReaderMRC(os.path.join(path, fname)) as reader:
    header = reader.read_header()
    data_arr = reader.read_data()

angles = np.deg2rad(
    [-48.0000, -44.0000, -42.0000, -40.0000, -38.0000, -36.0000, -34.0000,
     -32.0000, -26.0000, -24.0000, -17.0000, -14.0000, -12.0000, -10.0000,
     -8.00000, 1.00000, 3.00000, 6.00000, 9.00000, 12.0000, 15.0000, 18.0000,
     21.0000, 24.0000, 27.0000, 30.0000, 33.0000, 36.0000, 40.0000, 43.0000,
     46.0000])
angle_part = odl.nonuniform_partition(angles, min_pt=angles[0],
                                      max_pt=angles[-1])

num_angles = reader.data_shape[0]
assert num_angles == angles.size

det_shape = reader.data_shape[1:]
det_width = np.array(det_shape, dtype=float)
det_part = odl.uniform_partition(-det_width / 2, det_width / 2, det_shape)

geom = odl.tomo.Parallel3dAxisGeometry(angle_part, det_part)

# Do some crude data rescaling and cropping
win_size = 50

# Subtract the background means
win_pos = [det_shape[0] // 4, det_shape[1] // 4]
win_slices = (np.s_[win_pos[0] - win_size // 2: win_pos[0] + win_size // 2],
              np.s_[win_pos[1] - win_size // 2: win_pos[1] + win_size // 2])
bg_means = np.mean(data_arr[:, win_slices[0], win_slices[1]], axis=(1, 2))
rescaled_data_arr = data_arr - bg_means[:, None, None]

# Divide by mean of some area with an object part inside
win_pos = [det_shape[0] // 2, det_shape[1] // 2]
win_slices = (np.s_[win_pos[0] - win_size // 2: win_pos[0] + win_size // 2],
              np.s_[win_pos[1] - win_size // 2: win_pos[1] + win_size // 2])
obj_means = np.mean(rescaled_data_arr[:, win_slices[0], win_slices[1]],
                    axis=(1, 2))
rescaled_data_arr /= obj_means[:, None, None]

rescaled_data_arr[rescaled_data_arr <= 0] = 0
rescaled_data_arr[rescaled_data_arr >= 1.5] = 1.5


vol_shape = (256, 256, 256)
vol_extent = np.array(det_shape + (det_shape[0],), dtype=float) / np.sqrt(2)
reco_space = odl.uniform_discr(-vol_extent / 2, vol_extent / 2, vol_shape)
ray_trafo = odl.tomo.RayTransform(reco_space, geom)

data = ray_trafo.range.element(rescaled_data_arr)
proj_idx = 0
data.show('Projection image {}'.format(proj_idx),
          indices=[proj_idx, slice(None), slice(None)], clim=[0, 1.5])
sino_idx = det_shape[0] // 3
data.show('Sinogram {}'.format(sino_idx),
          indices=[slice(None), slice(None), sino_idx])


fbp_op = odl.tomo.fbp_op(ray_trafo, filter_type='Hamming', padding=True,
                         frequency_scaling=0.8)
fbp_reco = fbp_op(data)
fbp_reco.show()
