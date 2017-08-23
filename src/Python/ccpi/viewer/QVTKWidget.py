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

class QVTKWidget(QtWidgets.QWidget):

    """ A QVTKWidget for Python and Qt."""

    # Map between VTK and Qt cursors.
    _CURSOR_MAP = {
        0:  QtCore.Qt.ArrowCursor,          # VTK_CURSOR_DEFAULT
        1:  QtCore.Qt.ArrowCursor,          # VTK_CURSOR_ARROW
        2:  QtCore.Qt.SizeBDiagCursor,      # VTK_CURSOR_SIZENE
        3:  QtCore.Qt.SizeFDiagCursor,      # VTK_CURSOR_SIZENWSE
        4:  QtCore.Qt.SizeBDiagCursor,      # VTK_CURSOR_SIZESW
        5:  QtCore.Qt.SizeFDiagCursor,      # VTK_CURSOR_SIZESE
        6:  QtCore.Qt.SizeVerCursor,        # VTK_CURSOR_SIZENS
        7:  QtCore.Qt.SizeHorCursor,        # VTK_CURSOR_SIZEWE
        8:  QtCore.Qt.SizeAllCursor,        # VTK_CURSOR_SIZEALL
        9:  QtCore.Qt.PointingHandCursor,   # VTK_CURSOR_HAND
        10: QtCore.Qt.CrossCursor,          # VTK_CURSOR_CROSSHAIR
    }

    def __init__(self, parent=None, wflags=QtCore.Qt.WindowFlags(), **kw):
        # the current button
        self._ActiveButton = QtCore.Qt.NoButton

        # private attributes
        self.__oldFocus = None
        self.__saveX = 0
        self.__saveY = 0
        self.__saveModifiers = QtCore.Qt.NoModifier
        self.__saveButtons = QtCore.Qt.NoButton
        self.__timeframe = 0

        # create qt-level widget
        QtWidgets.QWidget.__init__(self, parent, wflags|QtCore.Qt.MSWindowsOwnDC)
        
        # Link to PyVE Viewer
        self._PyveViewer = CILViewer2D()
        #self._Viewer = self._PyveViewer._vtkPyveViewer
        
        self._Iren = self._PyveViewer.GetInteractor()
        #self._Iren = self._Viewer.GetRenderWindow().GetInteractor()
        self._RenderWindow = self._PyveViewer.GetRenderWindow()
        #self._RenderWindow = self._Viewer.GetRenderWindow()
        
        self._Iren.Register(self._RenderWindow)
        self._Iren.SetRenderWindow(self._RenderWindow)
        self._RenderWindow.SetWindowInfo(str(int(self.winId())))

        # do all the necessary qt setup
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        self.setAttribute(QtCore.Qt.WA_PaintOnScreen)
        self.setMouseTracking(True) # get all mouse events
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))

        self._Timer = QtCore.QTimer(self)
        #self.connect(self._Timer, QtCore.pyqtSignal('timeout()'), self.TimerEvent)

        self._Iren.AddObserver('CreateTimerEvent', self.CreateTimer)
        self._Iren.AddObserver('DestroyTimerEvent', self.DestroyTimer)
        self._Iren.GetRenderWindow().AddObserver('CursorChangedEvent',
                                                 self.CursorChangedEvent)

    # Destructor
    def __del__(self):
        self._Iren.UnRegister(self._RenderWindow)
        #QtWidgets.QWidget.__del__(self)

    # Display image data
    def SetInput(self, imageData):
        self._PyveViewer.setInput3DData(imageData)
        
    # GetInteractor
    def GetInteractor(self):
        return self._Iren
    
    # Display image data
    def GetPyveViewer(self):
        return self._PyveViewer

    def __getattr__(self, attr):
        """Makes the object behave like a vtkGenericRenderWindowInteractor"""
        print (attr)
        if attr == '__vtk__':
            return lambda t=self._Iren: t
        elif hasattr(self._Iren, attr):
            return getattr(self._Iren, attr)
#        else:
#            raise AttributeError( self.__class__.__name__ + \
#                  " has no attribute named " + attr )

    def CreateTimer(self, obj, evt):
        self._Timer.start(10)

    def DestroyTimer(self, obj, evt):
        self._Timer.stop()
        return 1

    def TimerEvent(self):
        self._Iren.InvokeEvent("TimerEvent")

    def CursorChangedEvent(self, obj, evt):
        """Called when the CursorChangedEvent fires on the render window."""
        # This indirection is needed since when the event fires, the current
        # cursor is not yet set so we defer this by which time the current
        # cursor should have been set.
        QtCore.QTimer.singleShot(0, self.ShowCursor)

    def HideCursor(self):
        """Hides the cursor."""
        self.setCursor(QtCore.Qt.BlankCursor)

    def ShowCursor(self):
        """Shows the cursor."""
        vtk_cursor = self._Iren.GetRenderWindow().GetCurrentCursor()
        qt_cursor = self._CURSOR_MAP.get(vtk_cursor, QtCore.Qt.ArrowCursor)
        self.setCursor(qt_cursor)

    def sizeHint(self):
        return QtCore.QSize(400, 400)

    def paintEngine(self):
        return None

    def paintEvent(self, ev):
        self._RenderWindow.Render()

    def resizeEvent(self, ev):
        self._RenderWindow.Render()
        w = self.width()
        h = self.height()

        self._RenderWindow.SetSize(w, h)
        self._Iren.SetSize(w, h)

    def _GetCtrlShiftAlt(self, ev):
        ctrl = shift = alt = False

        if hasattr(ev, 'modifiers'):
            if ev.modifiers() & QtCore.Qt.ShiftModifier:
                shift = True
            if ev.modifiers() & QtCore.Qt.ControlModifier:
                ctrl = True
            if ev.modifiers() & QtCore.Qt.AltModifier:
                alt = True
        else:
            if self.__saveModifiers & QtCore.Qt.ShiftModifier:
                shift = True
            if self.__saveModifiers & QtCore.Qt.ControlModifier:
                ctrl = True
            if self.__saveModifiers & QtCore.Qt.AltModifier:
                alt = True

        return ctrl, shift, alt

    def enterEvent(self, ev):
        if not self.hasFocus():
            self.__oldFocus = self.focusWidget()
            self.setFocus()

        ctrl, shift, alt = self._GetCtrlShiftAlt(ev)
        self._Iren.SetEventInformationFlipY(self.__saveX, self.__saveY,
                                            ctrl, shift, chr(0), 0, None)
        self._Iren.SetAltKey(alt)
        self._Iren.InvokeEvent("EnterEvent")

    def leaveEvent(self, ev):
        if self.__saveButtons == QtCore.Qt.NoButton and self.__oldFocus:
            self.__oldFocus.setFocus()
            self.__oldFocus = None

        ctrl, shift, alt = self._GetCtrlShiftAlt(ev)
        self._Iren.SetEventInformationFlipY(self.__saveX, self.__saveY,
                                            ctrl, shift, chr(0), 0, None)
        self._Iren.SetAltKey(alt)
        self._Iren.InvokeEvent("LeaveEvent")

    def mousePressEvent(self, ev):
        ctrl, shift, alt = self._GetCtrlShiftAlt(ev)
        repeat = 0
        if ev.type() == QtCore.QEvent.MouseButtonDblClick:
            repeat = 1
        self._Iren.SetEventInformationFlipY(ev.x(), ev.y(),
                                            ctrl, shift, chr(0), repeat, None)

        self._Iren.SetAltKey(alt)
        self._ActiveButton = ev.button()

        if self._ActiveButton == QtCore.Qt.LeftButton:
            self._Iren.InvokeEvent("LeftButtonPressEvent")
        elif self._ActiveButton == QtCore.Qt.RightButton:
            self._Iren.InvokeEvent("RightButtonPressEvent")
        elif self._ActiveButton == QtCore.Qt.MidButton:
            self._Iren.InvokeEvent("MiddleButtonPressEvent")

    def mouseReleaseEvent(self, ev):
        ctrl, shift, alt = self._GetCtrlShiftAlt(ev)
        self._Iren.SetEventInformationFlipY(ev.x(), ev.y(),
                                            ctrl, shift, chr(0), 0, None)
        self._Iren.SetAltKey(alt)

        if self._ActiveButton == QtCore.Qt.LeftButton:
            self._Iren.InvokeEvent("LeftButtonReleaseEvent")
        elif self._ActiveButton == QtCore.Qt.RightButton:
            self._Iren.InvokeEvent("RightButtonReleaseEvent")
        elif self._ActiveButton == QtCore.Qt.MidButton:
            self._Iren.InvokeEvent("MiddleButtonReleaseEvent")

    def mouseMoveEvent(self, ev):
        self.__saveModifiers = ev.modifiers()
        self.__saveButtons = ev.buttons()
        self.__saveX = ev.x()
        self.__saveY = ev.y()

        ctrl, shift, alt = self._GetCtrlShiftAlt(ev)
        self._Iren.SetEventInformationFlipY(ev.x(), ev.y(),
                                            ctrl, shift, chr(0), 0, None)
        self._Iren.SetAltKey(alt)
        self._Iren.InvokeEvent("MouseMoveEvent")

    def keyPressEvent(self, ev):
        ctrl, shift, alt = self._GetCtrlShiftAlt(ev)
        if ev.key() < 256:
            key = str(ev.text())
        else:
            key = chr(0)

        self._Iren.SetEventInformationFlipY(self.__saveX, self.__saveY,
                                            ctrl, shift, key, 0, None)
        self._Iren.SetAltKey(alt)
        self._Iren.InvokeEvent("KeyPressEvent")
        self._Iren.InvokeEvent("CharEvent")

    def keyReleaseEvent(self, ev):
        ctrl, shift, alt = self._GetCtrlShiftAlt(ev)
        if ev.key() < 256:
            key = chr(ev.key())
        else:
            key = chr(0)

        self._Iren.SetEventInformationFlipY(self.__saveX, self.__saveY,
                                            ctrl, shift, key, 0, None)
        self._Iren.SetAltKey(alt)
        self._Iren.InvokeEvent("KeyReleaseEvent")

    def wheelEvent(self, ev):
        print ("angleDeltaX %d" % ev.angleDelta().x())
        print ("angleDeltaY %d" % ev.angleDelta().y())
        if ev.angleDelta().y() >= 0:
            self._Iren.InvokeEvent("MouseWheelForwardEvent")
        else:
            self._Iren.InvokeEvent("MouseWheelBackwardEvent")

    def GetRenderWindow(self):
        return self._RenderWindow

    def Render(self):
        self.update()


def QVTKExample():    
    """A simple example that uses the QVTKWidget class."""

    # every QT app needs an app
    app = QtWidgets.QApplication(['PyVE QVTKWidget Example'])
    page_VTK = QtWidgets.QWidget()
    page_VTK.resize(500,500)
    layout = QtWidgets.QVBoxLayout(page_VTK)
    # create the widget
    widget = QVTKWidget(parent=None)
    layout.addWidget(widget)
    
    #reader = vtk.vtkPNGReader()
    #reader.SetFileName("F:\Diagnostics\Images\PyVE\VTKData\Data\camscene.png")
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName("C:\\Users\\ofn77899\\Documents\\GitHub\\CCPi-Simpleflex\\data\\head.mha")
    reader.Update()
    
    widget.SetInput(reader.GetOutput())
    
    # show the widget
    page_VTK.show()
    # start event processing
    app.exec_()

if __name__ == "__main__":
    QVTKExample()
