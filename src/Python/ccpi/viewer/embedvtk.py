# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 12:18:58 2017

@author: ofn77899
"""

#!/usr/bin/env python
 
import sys
import vtk
from PyQt5 import QtCore, QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import QVTKWidget2
 
class MainWindow(QtWidgets.QMainWindow):
 
    def __init__(self, parent = None):
        QtWidgets.QMainWindow.__init__(self, parent)
 
        self.frame = QtWidgets.QFrame()
 
        self.vl = QtWidgets.QVBoxLayout()
#        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        
        self.vtkWidget = QVTKWidget2.QVTKWidget(self.frame)
        self.iren = self.vtkWidget.GetInteractor()
        self.vl.addWidget(self.vtkWidget)
        
        
        
    
        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
#        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
# 
#        # Create source
#        source = vtk.vtkSphereSource()
#        source.SetCenter(0, 0, 0)
#        source.SetRadius(5.0)
# 
#        # Create a mapper
#        mapper = vtk.vtkPolyDataMapper()
#        mapper.SetInputConnection(source.GetOutputPort())
# 
#        # Create an actor
#        actor = vtk.vtkActor()
#        actor.SetMapper(mapper)
# 
#        self.ren.AddActor(actor)
# 
#        self.ren.ResetCamera()
# 
        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)
        reader = vtk.vtkMetaImageReader()
        reader.SetFileName("C:\\Users\\ofn77899\\Documents\\GitHub\\CCPi-Simpleflex\\data\\head.mha")
        reader.Update()
        
        self.vtkWidget.SetInput(reader.GetOutput())
        
        #self.vktWidget.Initialize()
        #self.vktWidget.Start()
        
        self.show()
        #self.iren.Initialize()
 
 
if __name__ == "__main__":
 
    app = QtWidgets.QApplication(sys.argv)
 
    window = MainWindow()
 
    sys.exit(app.exec_())