# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:34:49 2017

@author: ofn77899
"""

import h5py
import numpy

def getEntry(nx, location):
    for item in nx[location].keys():
        print (item)
        
filename = r'/home/ofn77899/Reconstruction/CCPi-FISTA_Reconstruction/demos/DendrData.h5'
nx = h5py.File(filename, "r")
#getEntry(nx, '/')
# I have exported the entries as children of /
entries = [entry for entry in nx['/'].keys()]
print (entries)

Sino3D = numpy.asarray(nx.get('/Sino3D'))
Weights3D = numpy.asarray(nx.get('/Weights3D'))
angSize = numpy.asarray(nx.get('/angSize'), dtype=int)[0]
angles_rad = numpy.asarray(nx.get('/angles_rad'))
recon_size = numpy.asarray(nx.get('/recon_size'), dtype=int)[0]
size_det = numpy.asarray(nx.get('/size_det'), dtype=int)[0]

slices_tot = numpy.asarray(nx.get('/slices_tot'), dtype=int)[0]

#from ccpi.viewer.CILViewer2D import CILViewer2D
#v = CILViewer2D()
#v.setInputAsNumpy(Weights3D)
#v.startRenderLoop()

import matplotlib.pyplot as plt
fig = plt.figure()

a=fig.add_subplot(1,1,1)
a.set_title('noise')
imgplot = plt.imshow(Weights3D[0].T)
plt.show()
