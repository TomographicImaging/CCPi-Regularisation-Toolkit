import astra
import numpy
import filefun


# read in the same data as the DemoRD2
angles = filefun.dlmread("DemoRD2/angles.csv")
darks_ar = filefun.dlmread("DemoRD2/darks_ar.csv", separator=",")
flats_ar = filefun.dlmread("DemoRD2/flats_ar.csv", separator=",")

if True:
    Sino3D = numpy.load("DemoRD2/Sino3D.npy")
else:
    sino = filefun.dlmread("DemoRD2/sino_01.csv", separator=",")
    a = map (lambda x:x, numpy.shape(sino))
    a.append(20)

    Sino3D = numpy.zeros(tuple(a), dtype="float")

    for i in range(1,numpy.shape(Sino3D)[2]+1):
        print("Read file DemoRD2/sino_%02d.csv" % i)
        sino = filefun.dlmread("DemoRD2/sino_%02d.csv" % i, separator=",")
        Sino3D.T[i-1] = sino.T
    
Weights3D = numpy.asarray(Sino3D, dtype="float")

##angles_rad = angles*(pi/180); % conversion to radians
##size_det = size(data_raw3D,1); % detectors dim
##angSize = size(data_raw3D, 2); % angles dim
##slices_tot = size(data_raw3D, 3); % no of slices
##recon_size = 950; % reconstruction size


angles_rad = angles * numpy.pi /180.
size_det, angSize, slices_tot = numpy.shape(Sino3D)
size_det, angSize, slices_tot = [int(i) for i in numpy.shape(Sino3D)]
recon_size = 950
Z_slices = 3;
det_row_count = Z_slices;

#proj_geom = astra_create_proj_geom('parallel3d', 1, 1,
# det_row_count, size_det, angles_rad);

detectorSpacingX = 1.0
detectorSpacingY = detectorSpacingX
proj_geom = astra.create_proj_geom('parallel3d',
                                            detectorSpacingX,
                                            detectorSpacingY,
                                            det_row_count,
                                            size_det,
                                            angles_rad)

#vol_geom = astra_create_vol_geom(recon_size,recon_size,Z_slices);
vol_geom = astra.create_vol_geom(recon_size,recon_size,Z_slices);

sino = numpy.zeros((size_det, angSize, slices_tot), dtype="float")

#weights = ones(size(sino));
weights = numpy.ones(numpy.shape(sino))

#####################################################################
## PowerMethod for Lipschitz constant

N = vol_geom['GridColCount']
x1 = numpy.random.rand(1,N,N)
#sqweight = sqrt(weights(:,:,1));
sqweight = numpy.sqrt(weights.T[0]).T
##proj_geomT = proj_geom;
proj_geomT = proj_geom.copy()
##proj_geomT.DetectorRowCount = 1;
proj_geomT['DetectorRowCount'] = 1
##vol_geomT = vol_geom;
vol_geomT = vol_geom.copy()
##vol_geomT.GridSliceCount = 1;
vol_geomT['GridSliceCount'] = 1

##[sino_id, y] = astra_create_sino3d_cuda(x1, proj_geomT, vol_geomT);

#sino_id, y = astra.create_sino3d_gpu(x1, proj_geomT, vol_geomT);
sino_id, y = astra.create_sino(x1, proj_geomT, vol_geomT);

##y = sqweight.*y;
##astra_mex_data3d('delete', sino_id);
        

