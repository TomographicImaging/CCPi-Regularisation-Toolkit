import astra
import numpy

detectorSpacingX = 1.0
detectorSpacingY = 1.0
det_row_count = 128
det_col_count = 128

angles_rad = numpy.asarray([i for i in range(360)], dtype=float) / 180. * numpy.pi

proj_geom = astra.creators.create_proj_geom('parallel3d',
                                            detectorSpacingX,
                                            detectorSpacingY,
                                            det_row_count,
                                            det_col_count,
                                            angles_rad)

image_size_x = 64
image_size_y = 64
image_size_z = 32

vol_geom = astra.creators.create_vol_geom(image_size_x,image_size_y,image_size_z)

x1 = numpy.random.rand(image_size_z,image_size_y,image_size_x)
sino_id, y = astra.creators.create_sino3d_gpu(x1, proj_geom, vol_geom)
