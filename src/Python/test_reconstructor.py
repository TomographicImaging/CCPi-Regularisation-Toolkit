# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:34:49 2017

@author: ofn77899
Based on DemoRD2.m
"""

import h5py
import numpy

from ccpi.fista.FISTAReconstructor import FISTAReconstructor
import astra

##def getEntry(nx, location):
##    for item in nx[location].keys():
##        print (item)
  
filename = r'/home/ofn77899/Reconstruction/CCPi-FISTA_Reconstruction/demos/DendrData.h5'
nx = h5py.File(filename, "r")
#getEntry(nx, '/')
# I have exported the entries as children of /
entries = [entry for entry in nx['/'].keys()]
print (entries)

Sino3D = numpy.asarray(nx.get('/Sino3D'), dtype="float32")
Weights3D = numpy.asarray(nx.get('/Weights3D'), dtype="float32")
angSize = numpy.asarray(nx.get('/angSize'), dtype=int)[0]
angles_rad = numpy.asarray(nx.get('/angles_rad'), dtype="float32")
recon_size = numpy.asarray(nx.get('/recon_size'), dtype=int)[0]
size_det = numpy.asarray(nx.get('/size_det'), dtype=int)[0]
slices_tot = numpy.asarray(nx.get('/slices_tot'), dtype=int)[0]

Z_slices = 3
det_row_count = Z_slices
# next definition is just for consistency of naming
det_col_count = size_det

detectorSpacingX = 1.0
detectorSpacingY = detectorSpacingX


proj_geom = astra.creators.create_proj_geom('parallel3d',
                                            detectorSpacingX,
                                            detectorSpacingY,
                                            det_row_count,
                                            det_col_count,
                                            angles_rad)

#vol_geom = astra_create_vol_geom(recon_size,recon_size,Z_slices);
image_size_x = recon_size
image_size_y = recon_size
image_size_z = Z_slices
vol_geom = astra.creators.create_vol_geom( image_size_x,
                                           image_size_y,
                                           image_size_z)

## First pass the arguments to the FISTAReconstructor and test the
## Lipschitz constant

fistaRecon = FISTAReconstructor(proj_geom,
                                vol_geom,
                                Sino3D ,
                                weights=Weights3D)

print ("Lipschitz Constant {0}".format(fistaRecon.pars['Lipschitz_constant']))

## the calculation of the lipschitz constant should not start by itself
