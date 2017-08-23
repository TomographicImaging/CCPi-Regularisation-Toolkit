################################################################################
# File:         QVTKWidget.py
# Author:       Edoardo Pasca
# Description:  PyVE Viewer Qt widget
#
# License:
#               This file is part of PyVE. PyVE is an open-source image 
#               analysis and visualization environment focused on medical
#               imaging. More info at http://pyve.sourceforge.net
#	       
#               Copyright (c) 2011-2012 Edoardo Pasca, Lukas Batteau 
#               All rights reserved.
#	       
#              Redistribution and use in source and binary forms, with or
#              without modification, are permitted provided that the following
#              conditions are met:
#
#              Redistributions of source code must retain the above copyright
#              notice, this list of conditions and the following disclaimer.
#              Redistributions in binary form must reproduce the above
#              copyright notice, this list of conditions and the following
#              disclaimer in the documentation and/or other materials provided
#              with the distribution.  Neither name of Edoardo Pasca or Lukas
#              Batteau nor the names of any contributors may be used to endorse
#              or promote products derived from this software without specific
#              prior written permission.
#
#              THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
#              CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,
#              INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#              MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#              DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE
#              LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
#              OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#              PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
#              OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#              THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
#              TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
#              OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
#              OF SUCH DAMAGE.
#
# CHANGE HISTORY
#
# 20120118    Edoardo Pasca        Initial version
#             
###############################################################################

import os
from PyQt5 import QtCore, QtGui, QtWidgets
#import itk
import vtk
#from viewer import PyveViewer
from ccpi.viewer.CILViewer2D import CILViewer2D , Converter
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class QVTKWidget(QVTKRenderWindowInteractor):

    
    def __init__(self, parent=None, wflags=QtCore.Qt.WindowFlags(), **kw):
        kw = dict() 
        super().__init__(parent, **kw)
        
        
        # Link to PyVE Viewer
        self._PyveViewer = CILViewer2D(400,400)
        #self._Viewer = self._PyveViewer._vtkPyveViewer
        
        self._Iren = self._PyveViewer.GetInteractor()
        kw['iren'] = self._Iren
        #self._Iren = self._Viewer.GetRenderWindow().GetInteractor()
        self._RenderWindow = self._PyveViewer.GetRenderWindow()
        #self._RenderWindow = self._Viewer.GetRenderWindow()
        kw['rw'] = self._RenderWindow
       
        
        
       
    def GetInteractor(self):
        return self._Iren
    
    # Display image data
    def SetInput(self, imageData):
        self._PyveViewer.setInput3DData(imageData)
    