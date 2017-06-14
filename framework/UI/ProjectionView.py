#!/usr/bin/env python

import numpy as np

from qtpy.QtCore import Qt, QSize, QPoint
from qtpy.QtWidgets import QGridLayout, QVBoxLayout, QFileDialog
from qtpy.QtGui import QImage, QPalette, QColor
from qtpy.QtWidgets import QMenu, QLabel, QComboBox, QCheckBox, QSplitter, QGroupBox, QSlider
from PyQt5.QtOpenGL import QGLWidget
from qtpy.QtSvg import QSvgGenerator

# PyOpenGL imports
from OpenGL.GL import *
import OpenGL.arrays.vbo as glvbo
import OpenGL.GLUT as glut
import OpenGL.GLU as glu

from .BaseTopologicalView import BaseTopologicalView
from .GLSL import *

import math
from . import colors
import matplotlib.pyplot
from matplotlib import cm
import os

class GLPlotWidget(QGLWidget):
    def __init__(self):
        super(GLPlotWidget,self).__init__()
        self.shininess=45.
        self.lightPos = [2.,0.,2.,1.]
        self.setFocusPolicy(Qt.StrongFocus)

        self.xRot = 0
        self.yRot = 0
        self.zRot = 0

        self.bgColor = self.palette().color(QPalette.Background)
        self.gridColor = QColor(0,90,151)
        self.boxColor = QColor(255,255,255)
        self.lastPos = QPoint()
        self.positions = np.array([(.25,.25,.25),(.5,.5,.5),(.75,.75,.75),(.25,.75,.5)],dtype=np.float32)
        self.colors = np.array([(.25,.25,.25,.9),(.5,.5,.5,.9),(.75,.75,.75,.9),(.25,.75,.5,.9)],dtype=np.float32)
        self.points = []
        self.lines = []
        self.alphaTransparency = 0.2

    def xRotation(self):
        return self.xRot

    def yRotation(self):
        return self.yRot

    def zRotation(self):
        return self.zRot

    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(400, 400)

    def setXRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.xRot:
            self.xRot = angle
            self.updateGL()

    def setYRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.yRot:
            self.yRot = angle
            self.updateGL()

    def setZRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.zRot:
            self.zRot = angle
            self.updateGL()

    def initializeGL(self):
        glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGBA | glut.GLUT_DEPTH | glut.GLUT_MULTISAMPLE)

        #VBO Stuff
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        # self.vbo = glvbo.VBO(self.positions)
        # self.vbo.bind()
        # glVertexPointer(3, GL_FLOAT, 0, self.positions)

        ##Aliasing
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POLYGON_SMOOTH)

        ## Depth/Alpha/Culing
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glEnable(GL_CULL_FACE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthFunc(GL_LESS)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glFrontFace(GL_CCW)

        ## Lighting
        glShadeModel(GL_SMOOTH)
        glEnable(GL_COLOR_MATERIAL)
        glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, [GL_TRUE])
        # glEnable(GL_LIGHTING)
        # glEnable(GL_LIGHT0)

        # mat_specular = [1.0, 1.0, 1.0, 1.0]
        # mat_ambient = [0.75, 0.75, 0.75, 0.75]
        # glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular)
        # glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, mat_ambient)

        # light_ambient0 = [0.5, 0.5, 0.5, 1.0]
        # light_diffuse0 = [1.0, 1.0, 1.0, 1.0]
        # light_specular0 = [1.0, 1.0, 1.0, 1.0]
        # glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient0)
        # glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse0)
        # glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular0)

        self.qglClearColor(self.bgColor)

        ########################################################################
        ## Shader initialization
        ########################################################################
        # Sprite shader
        resourceLocation = os.path.join(os.path.dirname(os.path.abspath(__file__)),'resources')

        vertexShaderFile = os.path.join(resourceLocation,'shaders','sprite.vert')
        vertexShader = open(vertexShaderFile)
        VS = vertexShader.read()
        vertexShader.close()
        fragmentShaderFile = os.path.join(resourceLocation,'shaders','sprite.frag')
        fragmentShader = open(fragmentShaderFile)
        FS = fragmentShader.read()
        fragmentShader.close()
        vs = compile_shader(VS,GL_VERTEX_SHADER)
        fs = compile_shader(FS,GL_FRAGMENT_SHADER)
        self.spriteProgramID = create_shader_program([vs,fs])
        uniform_point_size = glGetUniformLocation(self.spriteProgramID, "point_size")
        glUseProgram(self.spriteProgramID)
        glUniform1f(uniform_point_size, 5)
        glUseProgram(0)

        # Depth shader
        # vertexShader = open('resources/shaders/DepthRTT.vertexshader')
        # VS = vertexShader.read()
        # vertexShader.close()
        # fragmentShader = open('resources/shaders/DepthRTT.fragmentshader')
        # FS = fragmentShader.read()
        # fragmentShader.close()
        # vs = compile_shader(VS,GL_VERTEX_SHADER)
        # fs = compile_shader(FS,GL_FRAGMENT_SHADER)
        # self.depthProgramID = create_shader_program([vs,fs])

        # # Texture shader
        # vertexShader = open('resources/shaders/Passthrough.vertexshader')
        # VS = vertexShader.read()
        # vertexShader.close()
        # fragmentShader = open('resources/shaders/SimpleTexture.fragmentshader')
        # FS = fragmentShader.read()
        # fragmentShader.close()
        # vs = compile_shader(VS,GL_VERTEX_SHADER)
        # fs = compile_shader(FS,GL_FRAGMENT_SHADER)
        # self.quadProgramID = create_shader_program([vs,fs])

        # # Shadow mapping shader
        # vertexShader = open('resources/shaders/ShadowMapping.vertexshader')
        # VS = vertexShader.read()
        # vertexShader.close()
        # fragmentShader = open('resources/shaders/ShadowMapping.fragmentshader')
        # FS = fragmentShader.read()
        # fragmentShader.close()
        # vs = compile_shader(VS,GL_VERTEX_SHADER)
        # fs = compile_shader(FS,GL_FRAGMENT_SHADER)
        # self.programID = create_shader_program([vs,fs])

        # # My Shaders
        # vertexShader = open('resources/shaders/lambert.vert')
        # VS = vertexShader.read()
        # vertexShader.close()
        # fragmentShader = open('resources/shaders/lambert.frag')
        # FS = fragmentShader.read()
        # fragmentShader.close()
        # vs = compile_shader(VS,GL_VERTEX_SHADER)
        # fs = compile_shader(FS,GL_FRAGMENT_SHADER)
        # self.shaders_program = create_shader_program([vs,fs])

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslated(0.5, 0.5, 0.5)
        glRotated(self.xRot / 16.0, 1.0, 0.0, 0.0)
        glRotated(self.yRot / 16.0, 0.0, 1.0, 0.0)
        glRotated(self.zRot / 16.0, 0.0, 0.0, 1.0)
        glTranslated(-0.5, -0.5, -0.5)
        self.drawBoundingGrid()
        glColor3f(1., 0., 0.)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glEnable(GL_POINT_SPRITE)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
        # glDisable(GL_DEPTH_TEST)
        # glDrawElements(GL_POINTS, len(self.positions), GL_UNSIGNED_INT, None)
        glUseProgram(self.spriteProgramID)
        glVertexPointer(3, GL_FLOAT, 0, self.positions)
        glColorPointer(4, GL_FLOAT, 0, self.colors)
        glDrawElements(GL_POINTS,len(self.points), GL_UNSIGNED_INT, np.array(self.points, dtype=np.uint32))

        colorAlpha = np.array(self.colors)
        for row in range(len(colorAlpha)):
          colorAlpha[row][3] = self.alphaTransparency
        glColorPointer(4, GL_FLOAT, 0, colorAlpha)
        glDrawElements(GL_POINTS,len(self.positions), GL_UNSIGNED_INT, np.array(range(len(self.positions)), dtype=np.uint32))

        glUseProgram(0)
        glEnable(GL_DEPTH_TEST)
        if len(self.lines) > 0:
          glColorPointer(4, GL_FLOAT, 0, self.colors)
          glDrawElements(GL_LINES,2*len(self.lines), GL_UNSIGNED_INT, np.array(self.lines, dtype=np.uint32))
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    def resizeGL(self, width, height):
        side = min(width, height)
        glViewport((width - side) // 2, (height - side) // 2, side, side)

        eps = math.sqrt(2)/2.
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # glOrtho(-0.5, +0.5, +0.5, -0.5, 4.0, 15.0)
        glOrtho(0-eps, 1+eps, 1+eps, 0.0-eps, -1-eps, 1+eps)
        glMatrixMode(GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.lastPos = QPoint(event.pos())

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()
        dy = -dy

        if event.buttons() & Qt.LeftButton:
            self.setXRotation(self.xRot + 8 * dy)
            self.setYRotation(self.yRot + 8 * dx)
        elif event.buttons() & Qt.RightButton:
            self.setXRotation(self.xRot + 8 * dy)
            self.setZRotation(self.zRot + 8 * dx)

        self.lastPos = QPoint(event.pos())

    def drawBoundingGrid(self):
      #Three passes:
      # 1 - Solid masks
      # 2 - Bounding lines
      # 3 - Grid lines
      for i in range(3):
        if i == 0:
          glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
          glDepthRange(0.1,1.1)
          self.qglColor(self.boxColor)
          gridCount = 1
        else:
          if i == 2:
            gridCount = 1
            glLineWidth(2)
          else:
            gridCount = 4
            glLineWidth(1)
          glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
          glDepthRange(0.0,1.0)
          self.qglColor(self.gridColor)
        glBegin(GL_QUADS)
        for j in range(gridCount+1):
          valj = j/float(gridCount)
          for k in range(gridCount+1):
            valk = k/float(gridCount)
            glVertex3d(0,    0,    0)
            glVertex3d(0,    0, valk)
            glVertex3d(0, valj, valk)
            glVertex3d(0, valj,    0)

            glVertex3d(   0,    0, 1)
            glVertex3d(valj,    0, 1)
            glVertex3d(valj, valk, 1)
            glVertex3d(   0, valk, 1)

            glVertex3d(   0, 0,    0)
            glVertex3d(valj, 0,    0)
            glVertex3d(valj, 0, valk)
            glVertex3d(   0, 0, valk)

            glVertex3d(1, valj, valk)
            glVertex3d(1,    0, valk)
            glVertex3d(1,    0,    0)
            glVertex3d(1, valj,    0)

            glVertex3d(   0,    0, 0)
            glVertex3d(   0, valk, 0)
            glVertex3d(valj, valk, 0)
            glVertex3d(valj,    0, 0)

            glVertex3d(valj, 1, valk)
            glVertex3d(valj, 1, 0)
            glVertex3d(   0, 1, 0)
            glVertex3d(   0, 1, valk)
        glEnd()

      glDepthRange(0.0,1.0)
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle

class ProjectionView(BaseTopologicalView):
  def __init__(self, parent=None, amsc=None, title=None):
    super(ProjectionView, self).__init__(parent,amsc,title)

  def Reinitialize(self, parent=None, amsc=None, title=None):
    # Try to apply a new layout, if one already exists then make sure to grab
    # it for updating
    self.setLayout(QVBoxLayout())
    layout = self.layout()
    self.clearLayout(layout)

    mySplitter = QSplitter()
    mySplitter.setOrientation(Qt.Vertical)
    layout.addWidget(mySplitter)

    self.canvas = GLPlotWidget()

    mySplitter.addWidget(self.canvas)

    controls = QGroupBox()
    controls.setLayout(QGridLayout())
    subLayout = controls.layout()
    row = 0
    col = 0

    self.rightClickMenu = QMenu()
    self.axesLabelAction = self.rightClickMenu.addAction('Show Axis Labels')
    self.axesLabelAction.setCheckable(True)
    self.axesLabelAction.setChecked(True)
    self.axesLabelAction.triggered.connect(self.updateScene)

    self.chkExts = QCheckBox('Show Extrema')
    self.chkExts.setTristate(True)
    # self.chkExts.setCheckState(Qt.PartiallyChecked)
    self.chkExts.setCheckState(Qt.Unchecked)
    self.chkExts.setEnabled(False)
    self.chkExts.stateChanged.connect(self.updateScene)
    subLayout.addWidget(self.chkExts,row,col)
    row += 1
    col = 0

    self.chkEdges = QCheckBox('Show Edges')
    self.chkEdges.setChecked(False)
    self.chkEdges.stateChanged.connect(self.updateScene)
    subLayout.addWidget(self.chkEdges,row,col)
    row += 1
    col = 0

    self.cmbVars = {}
    for i,name in enumerate(['X','Y','Z','Color']):
      varLabel = name + ' variable:'
      self.cmbVars[name] = QComboBox()
      dimNames = self.amsc.GetNames()
      self.cmbVars[name].addItems(dimNames)
      if name == 'Color':
        self.cmbVars[name].addItems(['Segment'])
        self.cmbVars[name].addItems(['Minimum Flow'])
        self.cmbVars[name].addItems(['Maximum Flow'])
      self.cmbVars[name].addItem('Predicted from Linear Fit')
      self.cmbVars[name].addItem('Predicted from Minimum Fit')
      self.cmbVars[name].addItem('Predicted from Maximum Fit')
      self.cmbVars[name].addItem('Residual from Linear Fit')
      self.cmbVars[name].addItem('Residual from Minimum Fit')
      self.cmbVars[name].addItem('Residual from Maximum Fit')

      if i < len(dimNames):
        self.cmbVars[name].setCurrentIndex(i)
      else:
        self.cmbVars[name].setCurrentIndex(len(dimNames)-1)

      self.cmbVars[name].currentIndexChanged.connect(self.updateScene)

      subLayout.addWidget(QLabel(varLabel),row,col)
      col += 1
      subLayout.addWidget(self.cmbVars[name],row,col)
      row += 1
      col = 0

    self.cmbVars['Color'].currentIndexChanged.connect(self.updateColorVariable)

    self.cmbColorMaps = QComboBox()
    self.cmbColorMaps.addItems(matplotlib.pyplot.colormaps())
    self.cmbColorMaps.setCurrentIndex(self.cmbColorMaps.findText(colors.defaultCmap))
    self.cmbColorMaps.currentIndexChanged.connect(self.updateScene)
    subLayout.addWidget(QLabel('Colormap'),row,col)
    col += 1
    subLayout.addWidget(self.cmbColorMaps,row,col)
    row += 1
    col = 0
    subLayout.addWidget(QLabel('Context Alpha'), row, col)
    self.alphaSlider = QSlider()
    self.alphaSlider.setMinimum(0)
    self.alphaSlider.setMaximum(100)
    self.alphaSlider.setOrientation(Qt.Horizontal)
    self.alphaSlider.setSliderPosition(20)
    self.alphaSlider.valueChanged.connect(self.updateScene)

    subLayout.addWidget(self.alphaSlider)
    mySplitter.addWidget(controls)

    captureAction = self.rightClickMenu.addAction('Capture')
    captureAction.triggered.connect(self.saveImage)

    self.modelsChanged()
    self.updateScene()

  def saveImage(self):
    dialog = QFileDialog(self)
    dialog.setFileMode(QFileDialog.AnyFile)
    dialog.setAcceptMode(QFileDialog.AcceptSave)
    dialog.exec_()
    if dialog.result() == QFileDialog.Accepted:
      myFile = dialog.selectedFiles()[0]
      if myFile.endswith('.svg'):
        svgGen = QSvgGenerator()
        svgGen.setFileName(myFile)
        svgGen.setSize(self.canvas.size())
        svgGen.setTitle("Screen capture of " + self.__class__.__name__)
        # Potential name of this software: Quetzal: It is a pretty and colorful
        # and esoteric enough to interest people
        svgGen.setDescription("Generated from Quetzal.")
        painter = QPainter (svgGen)
      else:
        image = QImage(self.canvas.size(), QImage.Format_ARGB32)
        image.fill(Qt.transparent)
        painter = QPainter(image)
      self.canvas.render(painter)
      if myFile.endswith('.svg'):
        pass
      else:
        image.save(myFile,quality=100)
      del painter

  def updateColorVariable(self):
    pass

  def sizeHint(self):
    return QSize(300,600)

  def contextMenuEvent(self,event):
    pass
    #self.rightClickMenu.popup(event.globalPos())

  def update(self):
    super(ScatterView,self).update()

  def persistenceChanged(self):
    self.modelsChanged()

  def selectionChanged(self):
    self.updateScene()

  def modelsChanged(self):
    enabled = self.amsc.FitsSynced()
    for cmb in self.cmbVars.values():
      for i in range(cmb.count()):
        if 'Predicted' in cmb.itemText(i) or 'Residual' in cmb.itemText(i):
          item = cmb.model().item(i,0)
          if enabled:
            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
          else:
            item.setFlags(Qt.NoItemFlags)
        ## If this cmb is currently displaying fit information, then change it
        ## to display the output dimension
        if not enabled and (   'Predicted' in cmb.currentText() \
                            or 'Residual' in cmb.currentText()):
          cmb.setCurrentIndex(self.amsc.GetDimensionality())
    self.updateScene()

  def updateScene(self):
    rows = self.amsc.GetSelectedIndices()
    names = self.amsc.GetNames()

    myColormap = cm.get_cmap(self.cmbColorMaps.currentText())

    if len(rows) == 0:
      rows = list(range(self.amsc.GetSampleSize()))

    allValues = {}
    mins = {}
    maxs = {}
    minValues = {}
    maxValues = {}

    minIdxs = []
    maxIdxs = []

    if self.chkExts.checkState() == Qt.Checked \
    or self.chkExts.checkState() == Qt.PartiallyChecked:
      minMaxPairs = self.amsc.GetSelectedSegments()
      for extPair in minMaxPairs:
          minIdxs.append(extPair[0])
          maxIdxs.append(extPair[1])

      extIdxs = self.amsc.GetSelectedExtrema()
      for extIdx in extIdxs:
        if self.amsc.GetClassification(extIdx) == 'maximum':
          maxIdxs.append(extIdx)
        elif self.amsc.GetClassification(extIdx) == 'minimum':
          minIdxs.append(extIdx)

      ## Remove any duplicates
      minIdxs = list(set(minIdxs))
      maxIdxs = list(set(maxIdxs))

      if len(minIdxs) == 0 and len(maxIdxs) == 0:
        minMaxPairs = self.amsc.GetCurrentLabels()
        for extPair in minMaxPairs:
            minIdxs.append(extPair[0])
            maxIdxs.append(extPair[1])

      ## Remove the extrema from the list of regular points that will be
      ## rendered
      for extIdx in minIdxs + maxIdxs:
        if extIdx in rows:
            rows.remove(extIdx)

    specialColorKeywords = ['Segment','Minimum Flow', 'Maximum Flow']

    for key,cmb in self.cmbVars.items():
      if cmb.currentText() == 'Predicted from Linear Fit':
        allValues[key] = self.amsc.PredictY(None)
        mins[key] = min(allValues[key])
        maxs[key] = max(allValues[key])
        minValues[key] = allValues[key][minIdxs]
        maxValues[key] = allValues[key][maxIdxs]
      elif cmb.currentText() == 'Predicted from Minimum Fit':
        allValues[key] = self.amsc.PredictY(None,'minimum')
        mins[key] = min(allValues[key])
        maxs[key] = max(allValues[key])
        minValues[key] = allValues[key][minIdxs]
        maxValues[key] = allValues[key][maxIdxs]
      elif cmb.currentText() == 'Predicted from Maximum Fit':
        allValues[key] = self.amsc.PredictY(None,'maximum')
        mins[key] = min(allValues[key])
        maxs[key] = max(allValues[key])
        minValues[key] = allValues[key][minIdxs]
        maxValues[key] = allValues[key][maxIdxs]
      elif cmb.currentText() == 'Residual from Linear Fit':
        allValues[key] = self.amsc.Residuals(None)
        mins[key] = min(allValues[key])
        maxs[key] = max(allValues[key])
        minValues[key] = allValues[key][minIdxs]
        maxValues[key] = allValues[key][maxIdxs]
      elif cmb.currentText() == 'Residual from Minimum Fit':
        allValues[key] = self.amsc.Residuals(None,'minimum')
        mins[key] = min(allValues[key])
        maxs[key] = max(allValues[key])
        minValues[key] = allValues[key][minIdxs]
        maxValues[key] = allValues[key][maxIdxs]
      elif cmb.currentText() == 'Residual from Maximum Fit':
        allValues[key] = self.amsc.Residuals(None,'maximum')
        mins[key] = min(allValues[key])
        maxs[key] = max(allValues[key])
        minValues[key] = allValues[key][minIdxs]
        maxValues[key] = allValues[key][maxIdxs]
      elif cmb.currentText() == 'Segment':
        colorMap = self.amsc.GetColors()
        partitions = self.amsc.partitions
        allValues[key] = np.zeros(self.amsc.GetSampleSize(),dtype='|S7')
        for extPair,items in partitions.items():
          for item in items:
            allValues[key][item] = colorMap[extPair]
        minValues[key] = [colorMap[minIdx] for minIdx in minIdxs]
        maxValues[key] = [colorMap[maxIdx] for maxIdx in maxIdxs]
      elif cmb.currentText() == 'Maximum Flow':
        colorMap = self.amsc.GetColors()
        partitions = self.amsc.partitions
        allValues[key] = np.zeros(self.amsc.GetSampleSize(),dtype='|S7')
        for extPair,items in partitions.items():
          for item in items:
            allValues[key][item] = colorMap[extPair[1]]
        minValues[key] = [colorMap[minIdx] for minIdx in minIdxs]
        maxValues[key] = [colorMap[maxIdx] for maxIdx in maxIdxs]
      elif cmb.currentText() == 'Minimum Flow':
        colorMap = self.amsc.GetColors()
        partitions = self.amsc.partitions
        allValues[key] = np.zeros(self.amsc.GetSampleSize(),dtype='|S7')
        for extPair,items in partitions.items():
          for item in items:
            allValues[key][item] = colorMap[extPair[0]]
        minValues[key] = [colorMap[minIdx] for minIdx in minIdxs]
        maxValues[key] = [colorMap[maxIdx] for maxIdx in maxIdxs]
      else:
        col = names.index(cmb.currentText())
        if col == len(names)-1:
          allValues[key] = self.amsc.GetY(None)
          mins[key] = min(allValues[key])
          maxs[key] = max(allValues[key])
          minValues[key] = allValues[key][minIdxs]
          maxValues[key] = allValues[key][maxIdxs]
        else:
          allValues[key] = self.amsc.GetX(None,col)
          mins[key] = min(allValues[key])
          maxs[key] = max(allValues[key])
          minValues[key] = allValues[key][minIdxs]
          maxValues[key] = allValues[key][maxIdxs]

    Xs = (allValues['X'] - min(allValues['X'])) / (max(allValues['X']) - min(allValues['X']))
    Ys = (allValues['Y'] - min(allValues['Y'])) / (max(allValues['Y']) - min(allValues['Y']))
    Zs = (allValues['Z'] - min(allValues['Z'])) / (max(allValues['Z']) - min(allValues['Z']))
    if self.cmbVars['Color'].currentText() not in specialColorKeywords:
      Cs = (allValues['Color'] - min(allValues['Color'])) / (max(allValues['Color']) - min(allValues['Color']))
      Cs = np.array(myColormap(Cs))
    else:
      Cs = np.array([ QColor(val).getRgbF() for val in allValues['Color']])
    self.canvas.positions = np.transpose(np.vstack((Xs, max(Zs)-Zs, max(Ys)-Ys)))
    self.canvas.colors = Cs
    self.canvas.points = rows
    self.canvas.alphaTransparency = float(self.alphaSlider.sliderPosition()) / 100.
    if self.chkEdges.isChecked():
      lines  = []
      lineColors = []
      for row in rows:
        cols = self.amsc.GetNeighbors(row)
        for col in cols:
          if col in rows:
            lines.append((row,col))
            # if self.cmbVars['Color'].currentText() not in specialColorKeywords:
            #   lineColors.append(myColormap(((allValues['Color'][row]+allValues['Color'][col])/2.-mins['Color'])/(maxs['Color']-mins['Color'])))
            # elif allValues['Color'][row] == allValues['Color'][col]:
            #   lineColors.append(allValues['Color'][row])
            # else:
            #   lineColors.append('#CCCCCC')
      self.canvas.lines = lines
      # self.canvas.lineColors = lineColors
    else:
      self.canvas.lines = []
    if self.cmbVars['Color'].currentText() not in specialColorKeywords:
      if self.chkExts.checkState() == Qt.PartiallyChecked:
        maxValues['Color'] = 'red'
        minValues['Color'] = 'blue'
    else:
      if self.chkExts.checkState() == Qt.PartiallyChecked:
        maxValues['Color'] = 'red'
        minValues['Color'] = 'blue'

    self.canvas.update()