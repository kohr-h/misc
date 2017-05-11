import numpy as np
import pickle
import odl

# Define reco space
vol_size = np.array([230.0, 230.0])
vol_min = np.array([-115.0, -115.0])
shape = (512, 512)
reco_space = odl.uniform_discr(vol_min, vol_min + vol_size, shape)

# Set paths and file names
data_path = '/home/hkohr/SciData/Head_CT_Sim/'
geom_fname = 'HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_2D.geometry.p'
data_fname = 'HelicalSkullCT_70100644Phantom_no_bed_Dose150mGy_2D_120kV.npy'

# Get geometry and data
with open(data_path + geom_fname, 'rb') as f:
    geom = pickle.load(f, encoding='latin1')

data_arr = np.load(data_path + data_fname).astype('float32')
log_data_arr = -np.log(data_arr / np.max(data_arr))

# Define ray transform
ray_trafo = odl.tomo.RayTransform(reco_space, geom)

# Initialize data as ODL space element and display it, clipping to a
# somewhat reasonable range
data = ray_trafo.range.element(log_data_arr)
data.show('Sinogram', clim=[0, 4.5])

#  Compute FBP reco for a good initial guess
fbp = odl.tomo.fbp_op(ray_trafo, padding=True, filter_type='Hamming',
                      frequency_scaling=0.8)
fbp_reco = fbp(data)
fbp_reco.show('FBP reconstruction', clim=[0.019, 0.023])
