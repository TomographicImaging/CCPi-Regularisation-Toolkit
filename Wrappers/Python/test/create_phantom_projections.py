from ccpi.reconstruction.AstraDevice import AstraDevice
from ccpi.reconstruction.DeviceModel import DeviceModel
import h5py
import numpy
import matplotlib.pyplot as plt

nx = h5py.File('phant3D_256.h5', "r")
phantom = numpy.asarray(nx.get('/dataset1'))
pX,pY,pZ = numpy.shape(phantom)

filename = r'/home/ofn77899/Reconstruction/CCPi-FISTA_Reconstruction/demos/DendrData.h5'
nxa = h5py.File(filename, "r")
#getEntry(nx, '/')
# I have exported the entries as children of /
entries = [entry for entry in nxa['/'].keys()]
print (entries)

angles_rad = numpy.asarray(nxa.get('/angles_rad'), dtype="float32")


device = AstraDevice(
    DeviceModel.DeviceType.PARALLEL3D.value,
    [ pX , pY , 1., 1., angles_rad],
    [ pX, pY, pZ ] )


proj = device.doForwardProject(phantom)
stack = [proj[:,i,:] for i in range(len(angles_rad))]
stack = numpy.asarray(stack)


fig = plt.figure()
a=fig.add_subplot(1,2,1)
a.set_title('proj')
imgplot = plt.imshow(proj[:,100,:])
a=fig.add_subplot(1,2,2)
a.set_title('stack')
imgplot = plt.imshow(stack[100])
plt.show()

pf = h5py.File("phantom3D256_projections.h5" , "w")
pf.create_dataset("/projections", data=stack)
pf.create_dataset("/sinogram", data=proj)
pf.create_dataset("/angles", data=angles_rad)
pf.create_dataset("/reconstruction_volume" , data=numpy.asarray([pX, pY, pZ]))
pf.create_dataset("/camera/size" , data=numpy.asarray([pX , pY ]))
pf.create_dataset("/camera/spacing" , data=numpy.asarray([1.,1.]))
pf.flush()
pf.close()
