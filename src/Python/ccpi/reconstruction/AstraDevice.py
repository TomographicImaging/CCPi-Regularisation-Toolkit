import astra
from DeviceModel import DeviceModel

class AstraDevice(DeviceModel):
    '''Concrete class for Astra Device'''

    def __init__(self,
                 device_type,
                 data_aquisition_geometry,
                 reconstructed_volume_geometry):
        

        self.proj_geom = astra.creators.create_proj_geom(
            device_type,
            self.acquisition_data_geometry['detectorSpacingX'],
            self.acquisition_data_geometry['detectorSpacingX'],
            self.acquisition_data_geometry['cameraX'],
            self.acquisition_data_geometry['cameraY'],
            self.acquisition_data_geometry['angles'],
            angles_rad
            )

        self.vol_geom = astra.creators.create_vol_geom(
            self.reconstructed_volume_geometry['X'],
            self.reconstructed_volume_geometry['Y'],
            self.reconstructed_volume_geometry['Z']
            )

    def doForwardProject(self, volume):
        '''Forward projects the volume according to the device geometry

Uses Astra-toolbox
'''
        sino_id, y = astra.creators.create_sino3d_gpu(
            volume, self.proj_geom, self.vol_geom)
        astra.matlab.data3d('delete', sino_id)
        return y

    def doBackwardProject(self, projections):
        '''Backward projects the projections according to the device geometry

Uses Astra-toolbox
'''
        idx, volume = \
               astra.creators.create_backprojection3d_gpu(
                   projections,
                   self.proj_geom,
                   self.vol_geom)
        astra.matlab.data3d('delete', idx)
        return volume

    def createReducedDevice(self):
        return AstraDevice(self.proj_geom['type'],
                    {'detectorSpacingX' : self.proj_geom['DetectorSpacingX'] ,
                     'detectorSpacingY' : self.proj_geom['DetectorSpacingY'] ,
                     'cameraX' : self.proj_geom['DetectorColCount'] ,
                     'cameraY' : 1 ,
                     'angles' : self.proj_geom['ProjectionAngles'] } ,
                    {
                        'X' : self.vol_geom['GridColCount'],
                        'Y' : self.vol_geom['GridRowCount']
                        'Z' : 1} ) 

if __name__=="main":
    a = AstraDevice()


