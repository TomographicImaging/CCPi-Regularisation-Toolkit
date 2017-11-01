# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:34:49 2017

@author: ofn77899
Based on DemoRD2.m
"""

import h5py
import numpy

from ccpi.reconstruction.FISTAReconstructor import FISTAReconstructor
import astra
import matplotlib.pyplot as plt
from ccpi.imaging.Regularizer import Regularizer
from ccpi.reconstruction.AstraDevice import AstraDevice
from ccpi.reconstruction.DeviceModel import DeviceModel

def RMSE(signal1, signal2):
    '''RMSE Root Mean Squared Error'''
    if numpy.shape(signal1) == numpy.shape(signal2):
        err = (signal1 - signal2)
        err = numpy.sum( err * err )/numpy.size(signal1);  # MSE
        err = sqrt(err);                                   # RMSE
        return err
    else:
        raise Exception('Input signals must have the same shape')
  
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

Z_slices = 20
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


## Create a Acquisition Device Model
## Must specify some parameters of the acquisition:

astradevice = AstraDevice(
                DeviceModel.DeviceType.PARALLEL3D.value,
                [det_row_count , det_col_count ,
                 detectorSpacingX, detectorSpacingY ,
                 angles_rad
                 ],
                [ image_size_x, image_size_y, image_size_z ] )

fistaRecon = FISTAReconstructor(proj_geom,
                            vol_geom,
                            Sino3D ,
                            weights=Weights3D,
                            device=astradevice,
                            Lipschitz_constant = 767893952.0,
                            subsets = 8)

print("Reconstruction using FISTA-OS-PWLS without regularization...")
fistaRecon.setParameter(number_of_iterations = 18)

### adjust the regularization parameter
##lc = fistaRecon.getParameter('Lipschitz_constant')
##fistaRecon.getParameter('regularizer')\
##             .setParameter(regularization_parameter=5e6/lc)
fistaRecon.use_device = True
if False:
    fistaRecon.prepareForIteration()
    X = fistaRecon.iterate(numpy.load("../test/X.npy"))
    numpy.save("FISTA-OS-PWLS.npy",X)

## setup a regularizer algorithm
regul = Regularizer(Regularizer.Algorithm.FGP_TV)
regul.setParameter(regularization_parameter=5e6,
                   number_of_iterations=50,
                   tolerance_constant=1e-4,
                   TV_penalty=Regularizer.TotalVariationPenalty.isotropic)
if False:
    # adjust the regularization parameter
    lc = fistaRecon.getParameter('Lipschitz_constant')
    regul.setParameter(regularization_parameter=5e6/lc)
    fistaRecon.setParameter(regularizer=regul)
    fistaRecon.prepareForIteration()
    X = fistaRecon.iterate(numpy.load("../test/X.npy"))
    numpy.save("FISTA-OS-PWLS-TV.npy",X)

if False:
    # adjust the regularization parameter
    lc = fistaRecon.getParameter('Lipschitz_constant')
    regul.setParameter(regularization_parameter=5e6/lc)
    fistaRecon.setParameter(regularizer=regul)
    fistaRecon.setParameter(ring_lambda_R_L1=0.002, ring_alpha=21)
    fistaRecon.prepareForIteration()
    X = fistaRecon.iterate(numpy.load("../test/X.npy"))
    numpy.save("FISTA-OS-GH-TV.npy",X)

if True:
    # adjust the regularization parameter
    lc = fistaRecon.getParameter('Lipschitz_constant')
    regul.setParameter(
        algorithm=Regularizer.Algorithm.TGV_PD,
        regularization_parameter=0.5/lc,
        number_of_iterations=5)
    fistaRecon.setParameter(regularizer=regul)
    fistaRecon.setParameter(ring_lambda_R_L1=0.002, ring_alpha=21)
    fistaRecon.prepareForIteration()
    X = fistaRecon.iterate(numpy.load("../test/X.npy"))
    numpy.save("FISTA-OS-GH-TGV.npy",X)
    
