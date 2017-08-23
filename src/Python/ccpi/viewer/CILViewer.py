# -*- coding: utf-8 -*-
#   Copyright 2017 Edoardo Pasca
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
   
import vtk
import numpy
import math
from vtk.util import numpy_support

SLICE_ORIENTATION_XY = 2 # Z
SLICE_ORIENTATION_XZ = 1 # Y
SLICE_ORIENTATION_YZ = 0 # X



class CILViewer():
    '''Simple 3D Viewer based on VTK classes'''
    
    def __init__(self, dimx=600,dimy=600):
        '''creates the rendering pipeline'''
        
        # create a rendering window and renderer
        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.SetSize(dimx,dimy)
        self.renWin.AddRenderer(self.ren)

        # img 3D as slice
        self.img3D = None
        self.sliceno = 0
        self.sliceOrientation = SLICE_ORIENTATION_XY
        self.sliceActor = None
        self.voi = None
        self.wl = None
        self.ia = None
        self.sliceActorNo = 0
        # create a renderwindowinteractor
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)

        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(self.style)

        self.ren.SetBackground(.1, .2, .4)

        self.actors = {}
        self.iren.RemoveObservers('MouseWheelForwardEvent')
        self.iren.RemoveObservers('MouseWheelBackwardEvent')
        
        self.iren.AddObserver('MouseWheelForwardEvent', self.mouseInteraction, 1.0)
        self.iren.AddObserver('MouseWheelBackwardEvent', self.mouseInteraction, 1.0)

        self.iren.RemoveObservers('KeyPressEvent')
        self.iren.AddObserver('KeyPressEvent', self.keyPress, 1.0)
        
        
        self.iren.Initialize()

        

    def getRenderer(self):
        '''returns the renderer'''
        return self.ren

    def getRenderWindow(self):
        '''returns the render window'''
        return self.renWin

    def getInteractor(self):
        '''returns the render window interactor'''
        return self.iren

    def getCamera(self):
        '''returns the active camera'''
        return self.ren.GetActiveCamera()

    def createPolyDataActor(self, polydata):
        '''returns an actor for a given polydata'''
        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(polydata)
        else:
            mapper.SetInputData(polydata)
   
        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        #actor.GetProperty().SetOpacity(0.8)
        return actor

    def setPolyDataActor(self, actor):
        '''displays the given polydata'''
        
        self.ren.AddActor(actor)
        
        self.actors[len(self.actors)+1] = [actor, True]
        self.iren.Initialize()
        self.renWin.Render()

    def displayPolyData(self, polydata):
        self.setPolyDataActor(self.createPolyDataActor(polydata))
        
    def hideActor(self, actorno):
        '''Hides an actor identified by its number in the list of actors'''
        try:
            if self.actors[actorno][1]:
                self.ren.RemoveActor(self.actors[actorno][0])
                self.actors[actorno][1] = False
        except KeyError as ke:
            print ("Warning Actor not present")
        
    def showActor(self, actorno, actor = None):
        '''Shows hidden actor identified by its number in the list of actors'''
        try:
            if not self.actors[actorno][1]:
                self.ren.AddActor(self.actors[actorno][0])
                self.actors[actorno][1] = True
                return actorno
        except KeyError as ke:
            # adds it to the actors if not there already
            if actor != None:
                self.ren.AddActor(actor)
                self.actors[len(self.actors)+1] = [actor, True]
                return len(self.actors)

    def addActor(self, actor):
        '''Adds an actor to the render'''
        return self.showActor(0, actor)
            
        
    def saveRender(self, filename, renWin=None):
        '''Save the render window to PNG file'''
        # screenshot code:
        w2if = vtk.vtkWindowToImageFilter()
        if renWin == None:
            renWin = self.renWin
        w2if.SetInput(renWin)
        w2if.Update()
         
        writer = vtk.vtkPNGWriter()
        writer.SetFileName("%s.png" % (filename))
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()

    
    def startRenderLoop(self):
        self.iren.Start()


    def setupObservers(self, interactor):
        interactor.RemoveObservers('LeftButtonPressEvent')
        interactor.AddObserver('LeftButtonPressEvent', self.mouseInteraction)
        interactor.Initialize()

        
    def mouseInteraction(self, interactor, event):
        if event == 'MouseWheelForwardEvent':
            maxSlice = self.img3D.GetDimensions()[self.sliceOrientation]
            if (self.sliceno + 1 < maxSlice):
                self.hideActor(self.sliceActorNo)
                self.sliceno = self.sliceno + 1
                self.displaySliceActor(self.sliceno)
        else:
            minSlice = 0
            if (self.sliceno - 1 > minSlice):
                self.hideActor(self.sliceActorNo)
                self.sliceno = self.sliceno - 1
                self.displaySliceActor(self.sliceno)
                 

    def keyPress(self, interactor, event):
        #print ("Pressed key %s" % interactor.GetKeyCode())
        # Slice Orientation 
        if interactor.GetKeyCode() == "x":
            # slice on the other orientation
            self.sliceOrientation = SLICE_ORIENTATION_YZ
            self.sliceno = int(self.img3D.GetDimensions()[1] / 2)
            self.hideActor(self.sliceActorNo)
            self.displaySliceActor(self.sliceno)
        elif interactor.GetKeyCode() == "y":
            # slice on the other orientation
            self.sliceOrientation = SLICE_ORIENTATION_XZ
            self.sliceno = int(self.img3D.GetDimensions()[1] / 2)
            self.hideActor(self.sliceActorNo)
            self.displaySliceActor(self.sliceno)
        elif interactor.GetKeyCode() == "z":
            # slice on the other orientation
            self.sliceOrientation = SLICE_ORIENTATION_XY
            self.sliceno = int(self.img3D.GetDimensions()[2] / 2)
            self.hideActor(self.sliceActorNo)
            self.displaySliceActor(self.sliceno)
        if interactor.GetKeyCode() == "X":
            # Change the camera view point
            camera = vtk.vtkCamera()
            camera.SetFocalPoint(self.ren.GetActiveCamera().GetFocalPoint())
            camera.SetViewUp(self.ren.GetActiveCamera().GetViewUp())
            newposition = [i for i in self.ren.GetActiveCamera().GetFocalPoint()]
            newposition[SLICE_ORIENTATION_YZ] = math.sqrt(newposition[SLICE_ORIENTATION_XY] ** 2 + newposition[SLICE_ORIENTATION_XZ] ** 2) 
            camera.SetPosition(newposition)
            camera.SetViewUp(0,0,-1)
            self.ren.SetActiveCamera(camera)
            self.ren.ResetCamera()
            self.ren.Render()
            interactor.SetKeyCode("x")
            self.keyPress(interactor, event)
        elif interactor.GetKeyCode() == "Y":
             # Change the camera view point
            camera = vtk.vtkCamera()
            camera.SetFocalPoint(self.ren.GetActiveCamera().GetFocalPoint())
            camera.SetViewUp(self.ren.GetActiveCamera().GetViewUp())
            newposition = [i for i in self.ren.GetActiveCamera().GetFocalPoint()]
            newposition[SLICE_ORIENTATION_XZ] = math.sqrt(newposition[SLICE_ORIENTATION_XY] ** 2 + newposition[SLICE_ORIENTATION_YZ] ** 2) 
            camera.SetPosition(newposition)
            camera.SetViewUp(0,0,-1)
            self.ren.SetActiveCamera(camera)
            self.ren.ResetCamera()
            self.ren.Render()
            interactor.SetKeyCode("y")
            self.keyPress(interactor, event)
        elif interactor.GetKeyCode() == "Z":
             # Change the camera view point
            camera = vtk.vtkCamera()
            camera.SetFocalPoint(self.ren.GetActiveCamera().GetFocalPoint())
            camera.SetViewUp(self.ren.GetActiveCamera().GetViewUp())
            newposition = [i for i in self.ren.GetActiveCamera().GetFocalPoint()]
            newposition[SLICE_ORIENTATION_XY] = math.sqrt(newposition[SLICE_ORIENTATION_YZ] ** 2 + newposition[SLICE_ORIENTATION_XZ] ** 2) 
            camera.SetPosition(newposition)
            camera.SetViewUp(0,0,-1)
            self.ren.SetActiveCamera(camera)
            self.ren.ResetCamera()
            self.ren.Render()
            interactor.SetKeyCode("z")
            self.keyPress(interactor, event)
        else :
            print ("Unhandled event %s" % interactor.GetKeyCode())


        
    def setInput3DData(self, imageData):
        self.img3D = imageData

    def setInputAsNumpy(self, numpyarray):
        if (len(numpy.shape(numpyarray)) == 3):
            doubleImg = vtk.vtkImageData()
            shape = numpy.shape(numpyarray)
            doubleImg.SetDimensions(shape[0], shape[1], shape[2])
            doubleImg.SetOrigin(0,0,0)
            doubleImg.SetSpacing(1,1,1)
            doubleImg.SetExtent(0, shape[0]-1, 0, shape[1]-1, 0, shape[2]-1)
            #self.img3D.SetScalarType(vtk.VTK_UNSIGNED_SHORT, vtk.vtkInformation())
            doubleImg.AllocateScalars(vtk.VTK_DOUBLE,1)
            
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        doubleImg.SetScalarComponentFromDouble(
                            i,j,k,0, numpyarray[i][j][k])
        #self.setInput3DData( numpy_support.numpy_to_vtk(numpyarray) )
            # rescale to appropriate VTK_UNSIGNED_SHORT
            stats = vtk.vtkImageAccumulate()
            stats.SetInputData(doubleImg)
            stats.Update()
            iMin = stats.GetMin()[0]
            iMax = stats.GetMax()[0]
            scale = vtk.VTK_UNSIGNED_SHORT_MAX / (iMax - iMin)

            shiftScaler = vtk.vtkImageShiftScale ()
            shiftScaler.SetInputData(doubleImg)
            shiftScaler.SetScale(scale)
            shiftScaler.SetShift(iMin)
            shiftScaler.SetOutputScalarType(vtk.VTK_UNSIGNED_SHORT)
            shiftScaler.Update()
            self.img3D = shiftScaler.GetOutput()
            
    def displaySliceActor(self, sliceno = 0):
        self.sliceno = sliceno
        first = False
        
        self.sliceActor , self.voi, self.wl , self.ia = \
                        self.getSliceActor(self.img3D,
                                                 sliceno,
                                                 self.sliceActor,
                                                 self.voi,
                                                 self.wl,
                                                 self.ia)
        no = self.showActor(self.sliceActorNo, self.sliceActor)
        self.sliceActorNo = no
        
        self.iren.Initialize()
        self.renWin.Render()
        
        return self.sliceActorNo

                      
    def getSliceActor(self,
                      imageData ,
                      sliceno=0,
                      imageActor=None ,
                      voi=None,
                      windowLevel=None,
                      imageAccumulate=None):
        '''Slices a 3D volume and then creates an actor to be rendered'''
        if (voi==None):
            voi = vtk.vtkExtractVOI()
            #voi = vtk.vtkImageClip()
        voi.SetInputData(imageData)
        #select one slice in Z
        extent = [ i for i in self.img3D.GetExtent()]
        extent[self.sliceOrientation * 2] = sliceno
        extent[self.sliceOrientation * 2 + 1] = sliceno 
        voi.SetVOI(extent[0], extent[1],
                   extent[2], extent[3],
                   extent[4], extent[5])
        
        voi.Update()
        # set window/level for all slices
        if imageAccumulate == None:
            imageAccumulate = vtk.vtkImageAccumulate()
        
        if (windowLevel == None):
            windowLevel = vtk.vtkImageMapToWindowLevelColors()
            imageAccumulate.SetInputData(imageData)
            imageAccumulate.Update()
            cmax = imageAccumulate.GetMax()[0]
            cmin = imageAccumulate.GetMin()[0]
            windowLevel.SetLevel((cmax+cmin)/2)
            windowLevel.SetWindow(cmax-cmin)

        windowLevel.SetInputData(voi.GetOutput())
        windowLevel.Update()
            
        if imageActor == None:
            imageActor = vtk.vtkImageActor()
        imageActor.SetInputData(windowLevel.GetOutput())
        imageActor.SetDisplayExtent(extent[0], extent[1],
                   extent[2], extent[3],
                   extent[4], extent[5])
        imageActor.Update()
        return (imageActor , voi, windowLevel, imageAccumulate)


    # Set interpolation on
    def setInterpolateOn(self):
        self.sliceActor.SetInterpolate(True)
        self.renWin.Render()

    # Set interpolation off
    def setInterpolateOff(self):
        self.sliceActor.SetInterpolate(False)
        self.renWin.Render()