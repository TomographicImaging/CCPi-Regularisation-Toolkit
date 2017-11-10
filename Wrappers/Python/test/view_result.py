import numpy
from ccpi.viewer.CILViewer2D import *
import sys
#reader = vtk.vtkMetaImageReader()
#reader.SetFileName("X_out_os_s.mhd")
#reader.Update()

X = numpy.load(sys.argv[1])

v = CILViewer2D()
v.setInputAsNumpy(X)
v.startRenderLoop()
