from abc import ABCMeta, abstractmethod
from enum import Enum

class DeviceModel(metaclass=ABCMeta):
    '''Abstract class that defines the device for projection and backprojection

This class defines the methods that must be implemented by concrete classes.

    '''
    
    class DeviceType(Enum):
        '''Type of device
PARALLEL BEAM
PARALLEL BEAM 3D
CONE BEAM
HELICAL'''
        
        PARALLEL = 'parallel'
        PARALLEL3D = 'parallel3d'
        CONE_BEAM = 'cone-beam'
        HELICAL = 'helical'
        
    def __init__(self,
                 device_type,
                 data_aquisition_geometry,
                 reconstructed_volume_geometry):
        '''Initializes the class

Mandatory parameters are:
device_type from DeviceType Enum
data_acquisition_geometry: tuple (camera_X, camera_Y)
reconstructed_volume_geometry: tuple (dimX,dimY,dimZ)
'''
        self.device_geometry = device_type
        self.acquisition_data_geometry = {
            'cameraX':           data_aquisition_geometry[0],
            'cameraY':           data_aquisition_geometry[1],
            'detectorSpacingX' : data_aquisition_geometry[2],
            'detectorSpacingY' : data_aquisition_geometry[3],
            'angles' :           data_aquisition_geometry[4],}
        self.reconstructed_volume_geometry = {
            'X': reconstructed_volume_geometry[0] ,
            'Y': reconstructed_volume_geometry[1] ,
            'Z': reconstructed_volume_geometry[2] }

    @abstractmethod
    def doForwardProject(self, volume):
        '''Forward projects the volume according to the device geometry'''
        return NotImplemented

    
    @abstractmethod
    def doBackwardProject(self, projections):
        '''Backward projects the projections according to the device geometry'''
        return NotImplemented

    @abstractmethod
    def createReducedDevice(self):
        '''Create a Device to do forward/backward projections on 2D slices'''
        return NotImplemented
    

