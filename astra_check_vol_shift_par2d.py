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
import scipy.io

# Create a basic 256x256 square volume geometry
vol_geom = astra.create_vol_geom(256, 256)

# Create a parallel beam geometry with 180 angles between 0 and pi, and
# 384 detector pixels of width 1.
# For more details on available geometries, see the online help of the
# function astra_create_proj_geom .
proj_geom = astra.create_proj_geom('parallel', 1.0, 384,
                                   np.linspace(0, np.pi, 180, False))

# Load a 256x256 phantom image
P = scipy.io.loadmat('phantom.mat')['phantom256']

# Create a sinogram using the GPU.
# Note that the first time the GPU is accessed, there may be a delay
# of up to 10 seconds for initialization.
proj_id1 = astra.create_projector('cuda', proj_geom, vol_geom)
sinogram_id1, sinogram1 = astra.create_sino(P, proj_id1)

# Shift window in x direction
shift = -50
vol_geom['option']['WindowMinX'] += shift
vol_geom['option']['WindowMaxX'] += shift

proj_id2 = astra.create_projector('cuda', proj_geom, vol_geom)
sinogram_id2, sinogram2 = astra.create_sino(P, proj_id2)

# Shift window in y direction
vol_geom['option']['WindowMinX'] -= shift
vol_geom['option']['WindowMaxX'] -= shift
vol_geom['option']['WindowMinY'] += shift
vol_geom['option']['WindowMaxY'] += shift

proj_id3 = astra.create_projector('cuda', proj_geom, vol_geom)
sinogram_id3, sinogram3 = astra.create_sino(P, proj_id3)

pylab.gray()
pylab.figure()
pylab.imshow(P)
pylab.figure()
pylab.imshow(sinogram1)
pylab.figure()
pylab.imshow(sinogram2)
pylab.figure()
pylab.imshow(sinogram3)
pylab.show()

# Free memory
astra.data2d.delete(sinogram_id1)
astra.projector.delete(proj_id1)
astra.data2d.delete(sinogram_id2)
astra.projector.delete(proj_id2)
astra.data2d.delete(sinogram_id3)
astra.projector.delete(proj_id3)
