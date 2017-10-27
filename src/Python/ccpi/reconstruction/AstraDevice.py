import astra
from ccpi.reconstruction.DeviceModel import DeviceModel

class AstraDevice(DeviceModel):
    '''Concrete class for Astra Device'''

    def __init__(self,
                 device_type,
                 data_aquisition_geometry,
                 reconstructed_volume_geometry):
        
        super(AstraDevice, self).__init__(device_type,
                                          data_aquisition_geometry,
                                          reconstructed_volume_geometry)

        self.type = device_type
        self.proj_geom = astra.creators.create_proj_geom(
            device_type,
            self.acquisition_data_geometry['detectorSpacingX'],
            self.acquisition_data_geometry['detectorSpacingX'],
            self.acquisition_data_geometry['cameraX'],
            self.acquisition_data_geometry['cameraY'],
            self.acquisition_data_geometry['angles'],
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
        proj_geom = astra.creators.create_proj_geom(
            device_type,
            self.acquisition_data_geometry['detectorSpacingX'],
            self.acquisition_data_geometry['detectorSpacingX'],
            self.acquisition_data_geometry['cameraX'],
            1,
            self.acquisition_data_geometry['angles'],
            )

        vol_geom = astra.creators.create_vol_geom(
            self.reconstructed_volume_geometry['X'],
            self.reconstructed_volume_geometry['Y'],
            1
            )
        return AstraDevice(proj_geom['type'] ,
                           proj_geom,
                           vol_geom)
        

        
if __name__=="main":
    a = AstraDevice()


