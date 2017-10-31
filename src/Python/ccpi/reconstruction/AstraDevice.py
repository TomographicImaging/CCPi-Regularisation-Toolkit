import astra
from ccpi.reconstruction.DeviceModel import DeviceModel
import numpy

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
            self.acquisition_data_geometry['detectorSpacingY'],
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
              
        try:
            sino_id, y = astra.creators.create_sino3d_gpu(
                volume, self.proj_geom, self.vol_geom)
            astra.matlab.data3d('delete', sino_id)
            return y
        except Exception as e:
            print(e)
            print("Value Error: ", self.proj_geom, self.vol_geom)

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
    
    def createReducedDevice(self, proj_par={'cameraY' : 1} , vol_par={'Z':1}):
        '''Change the definition of the current device by changing some parameter

VERY RISKY'''
        for k,v in proj_par.items():
            if k in self.acquisition_data_geometry.keys():
                print ("Reduced device updating " , k , v)
                self.acquisition_data_geometry[k] = v
        print ("Reduced Device: ", self.acquisition_data_geometry)
        proj_geom =  [ 
            self.acquisition_data_geometry['cameraX'],
            self.acquisition_data_geometry['cameraY'],
            self.acquisition_data_geometry['detectorSpacingX'],
            self.acquisition_data_geometry['detectorSpacingY'],
            self.acquisition_data_geometry['angles']
            ]
        
        for k,v in vol_par.items():
            if k in self.reconstructed_volume_geometry.keys():
                print ("Reduced device updating " , k , v)
                self.reconstructed_volume_geometry[k] = v
        print ("Reduced Device: ",self.reconstructed_volume_geometry)
        
        vol_geom = [
            self.reconstructed_volume_geometry['X'],
            self.reconstructed_volume_geometry['Y'],
            self.reconstructed_volume_geometry['Z']
            ]
        return AstraDevice(self.type, proj_geom, vol_geom)
        

        
if __name__=="main":
    a = AstraDevice()


