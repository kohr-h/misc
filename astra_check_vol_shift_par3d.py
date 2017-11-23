# -----------------------------------------------------------------------
# Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
#            2013-2016, CWI, Amsterdam
#
# Contact: astra@uantwerpen.be
# Website: http://www.astra-toolbox.com/
#
# This file is part of the ASTRA Toolbox.
#
#
# The ASTRA Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The ASTRA Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------

import astra
import numpy as np
import pylab

vol_geom = astra.create_vol_geom(128, 128, 128)

angles = np.linspace(0, np.pi, 180, False)
proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 128, 192, angles)

# Create a simple hollow cube phantom
cube = np.zeros((128, 128, 128))
cube[17:113, 17:113, 17:113] = 1
cube[33:97, 33:97, 33:97] = 0

# Create projection data from this
proj_id1, proj_data1 = astra.create_sino3d_gpu(cube, proj_geom, vol_geom)

# Shift window in x direction
shift = -50
vol_geom['option']['WindowMinX'] += shift
vol_geom['option']['WindowMaxX'] += shift

proj_id2, proj_data2 = astra.create_sino3d_gpu(cube, proj_geom, vol_geom)

# Shift window in y direction
vol_geom['option']['WindowMinX'] -= shift
vol_geom['option']['WindowMaxX'] -= shift
vol_geom['option']['WindowMinY'] += shift
vol_geom['option']['WindowMaxY'] += shift

proj_id3, proj_data3 = astra.create_sino3d_gpu(cube, proj_geom, vol_geom)

# Display middle sinogram
pylab.gray()
pylab.figure()
pylab.imshow(proj_data1[64, :, :])
pylab.figure()
pylab.imshow(proj_data2[64, :, :])
pylab.figure()
pylab.imshow(proj_data3[64, :, :])
pylab.figure()
pylab.imshow(proj_data1[:, 0, :])
pylab.figure()
pylab.imshow(proj_data2[:, 0, :])
pylab.figure()
pylab.imshow(proj_data3[:, 0, :])


# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
astra.data3d.delete(proj_id1)
astra.data3d.delete(proj_id2)
astra.data3d.delete(proj_id3)
