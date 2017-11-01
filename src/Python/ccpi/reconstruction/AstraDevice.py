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
        print ("Astra device created:")
        print ("Camera : {0}x{1}".format(self.proj_geom['DetectorColCount'],
               self.proj_geom['DetectorRowCount']))
        print ("number of projections " , len(self.proj_geom['ProjectionAngles']))
        
        self.vol_geom = astra.creators.create_vol_geom(
            self.reconstructed_volume_geometry['X'],
            self.reconstructed_volume_geometry['Y'],
            self.reconstructed_volume_geometry['Z']
            )
        print ("Reconstruction volume:")
        print ("[{0},{1},{2}]".format(self.vol_geom['GridColCount'],
                                      self.vol_geom['GridRowCount'],
                                      self.vol_geom['GridSliceCount']))

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
        '''Create a new device based on the current device by changing some parameter

VERY RISKY'''
        acquisition_data_geometry = self.acquisition_data_geometry.copy()
        for k,v in proj_par.items():
            if k in acquisition_data_geometry.keys():
                acquisition_data_geometry[k] = v
        proj_geom =  [ 
            acquisition_data_geometry['cameraX'],
            acquisition_data_geometry['cameraY'],
            acquisition_data_geometry['detectorSpacingX'],
            acquisition_data_geometry['detectorSpacingY'],
            acquisition_data_geometry['angles']
            ]

        reconstructed_volume_geometry = self.reconstructed_volume_geometry.copy()
        for k,v in vol_par.items():
            if k in reconstructed_volume_geometry.keys():
                reconstructed_volume_geometry[k] = v
        
        vol_geom = [
            reconstructed_volume_geometry['X'],
            reconstructed_volume_geometry['Y'],
            reconstructed_volume_geometry['Z']
            ]
        return AstraDevice(self.type, proj_geom, vol_geom)
        

        
if __name__=="main":
    a = AstraDevice()


