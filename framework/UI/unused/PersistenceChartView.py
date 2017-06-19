#!/usr/bin/env python

from PySide import QtCore as qtc
from PySide import QtGui as qtg
from PySide import QtSvg as qts

from GenericView import GenericView
from AMSC_Object import AMSC_Object

import numpy as np
from colors import *

class PersistenceGraphicsView(qtg.QGraphicsView):
  def mousePressEvent(self, event):
    super(PersistenceGraphicsView, self).mousePressEvent(event)
    self.parent().mousePressEvent(event,True)

  def mouseMoveEvent(self, event):
    if event.buttons() != qtc.Qt.MiddleButton:
      super(PersistenceGraphicsView, self).mouseMoveEvent(event)
    else:
      self.parent().mouseMoveEvent(event,True)

  def mouseReleaseEvent(self, event):
    if event.button() != qtc.Qt.LeftButton:
      super(PersistenceGraphicsView, self).mouseReleaseEvent(event)
    else:
      self.parent().mouseReleaseEvent(event,True)


'''
    A view that shows the persistence hierarchy, that is, when minima and maxima
    are simplified, in this view we can control the persistence and see how many
    maxima, minima, or a combined count of both still exist at a specified level
    of simplification.
'''
class PersistenceChartView(GenericView):
  def __init__(self, parent=None, amsc=None, title=None):
    super(PersistenceChartView, self).__init__(parent,amsc,title)

  def Reinitialize(self,parent=None, amsc=None, title=None):
    # Try to apply a new layout, if one already exists then make sure to grab
    # it for updating
    self.setLayout(qtg.QVBoxLayout())
    layout = self.layout()
    self.clearLayout(layout)

    self.scene = qtg.QGraphicsScene()
    ## TODO: Maybe this one should fill the entire viewing space?
    self.scene.setSceneRect(0,0,100,100)
   # self.scene.setSceneRect(0,0,200,100)
    self.gView = PersistenceGraphicsView(self.scene)
    self.gView.setRenderHints(qtg.QPainter.Antialiasing |
                              qtg.QPainter.SmoothPixmapTransform)
    self.gView.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
    self.gView.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)

    self.rightClickMenu = qtg.QMenu()

    blankAction = self.rightClickMenu.addAction('Force Screen Update')
    blankAction.triggered.connect(self.blank)
    self.displayMenu = self.rightClickMenu.addMenu('Display')
    extActions = []
    extActions.append(self.displayMenu.addAction('Extrema'))
    extActions.append(self.displayMenu.addAction('Maxima'))
    extActions.append(self.displayMenu.addAction('Minima'))

    self.fillAction = self.rightClickMenu.addAction('Fill viewport')
    self.fillAction.setCheckable(True)
    self.fillAction.setChecked(True)
    self.fillAction.triggered.connect(self.updateScene)

    incAction = self.rightClickMenu.addAction('Increase Persistence')
    incAction.triggered.connect(self.segmentation.increaseLevel)
    decAction = self.rightClickMenu.addAction('Decrease Persistence')
    decAction.triggered.connect(self.segmentation.decreaseLevel)

    self.extGroup = qtg.QActionGroup(self.displayMenu)
    for act in extActions:
      act.setCheckable(True)
      self.extGroup.addAction(act)
    extActions[0].setChecked(True)
    self.extGroup.triggered.connect(self.updateScene)

    self.gView.scale(self.gView.width()/self.scene.width(),
                     self.gView.height()/self.scene.height())

    captureAction = self.rightClickMenu.addAction('Capture')
    captureAction.triggered.connect(self.saveImage)

    layout.addWidget(self.gView)
    self.updateScene()

  def saveImage(self):
    ## This view will do nothing if it is not given an AMSC object
    if not isinstance(self.segmentation,AMSC_Object):
      return
    dialog = qtg.QFileDialog(self)
    dialog.setFileMode(qtg.QFileDialog.AnyFile)
    dialog.setAcceptMode(qtg.QFileDialog.AcceptSave)
    dialog.exec_()
    if dialog.result() == qtg.QFileDialog.Accepted:
      myFile = dialog.selectedFiles()[0]
      self.scene.clearSelection()
      self.scene.setSceneRect(self.scene.itemsBoundingRect())
      if myFile.endswith('.svg'):
        svgGen = qts.QSvgGenerator()
        svgGen.setFileName(myFile)
        svgGen.setSize(self.scene.sceneRect().size().toSize())
        svgGen.setViewBox(self.scene.sceneRect())
        svgGen.setTitle("Screen capture of " + self.__class__.__name__)
        # Potential name of this software: Quetzal: It is a pretty and colorful
        # and esoteric enough to interest people
        svgGen.setDescription("Generated from Quetzal.")
        painter = qtg.QPainter (svgGen)
      else:
        image = qtg.QImage(self.scene.sceneRect().size().toSize(), qtg.QImage.Format_ARGB32)
        image.fill(qtc.Qt.transparent)
        painter = qtg.QPainter(image)
      self.scene.render(painter)
      if myFile.endswith('.svg'):
        pass
      else:
        image.save(myFile,quality=100)
      del painter

  def blank(self):
    self.updateScene()

  def mousePressEvent(self, event, fromChild=False):
    if fromChild:
      mousePt = self.gView.mapToScene(event.x(),event.y())
    else:
      pass
      # super(PersistenceChartView, self.mousePressEvent(event))

  def mouseMoveEvent(self, event, fromChild=False):
    if fromChild:
      pass
    else:
      pass
      # super(PersistenceChartView, self.mousePressEvent(event))

  def mouseReleaseEvent(self, event, fromChild=False):
    ## This view will do nothing if it is not given an AMSC object
    if not isinstance(self.segmentation,AMSC_Object):
      return
    if fromChild:
      mergeSequence = self.segmentation.GetMergeSequence()
      mousePt = self.gView.mapToScene(event.x(),event.y()).x()
      persistences = [p for idx,(parent,p) in mergeSequence.iteritems()]
      persistences = sorted(set(persistences))
      minP = 0
      maxP = max(persistences)
      width = self.scene.width()
      persistence = np.clip(mousePt/width*(maxP-minP) + minP, minP, maxP)
      # idx = len(persistences)-1
      # while persistences[idx] >= persistence and idx > 0:
      #   idx -= 1
      # self.segmentation.Level(persistences[idx])
      self.segmentation.Level(persistence)

      #### Debugging
      # partitions = self.segmentation.partitions
      # print('Partitions')
      # print('#######################')
      # print('#######################')
      # print('#######################')
      # print('#######################')
      # print('#######################')
      # count1 = 0
      # for key,value in partitions.iteritems():
      #   if key[0] != key[1] and len(value) > 2:
      #     print(key,len(value))
      #     count1 += 1
      # print('~~~~~~~~~~~~~~~~~~~~~~~')
      # count2 = 0
      # for key,value in partitions.iteritems():
      #   if key[0] != key[1] and len(value) == 2:
      #     print(key,len(value))
      #     count2 += 1
      # print('~~~~~~~~~~~~~~~~~~~~~~~')
      # count3 = 0
      # for key,value in partitions.iteritems():
      #   if key[0] == key[1]:
      #     print(key[0])
      #     count3 += 1
      # print('Non-empty Partitions',count1)
      # print('Empty Partitions',count2)
      # print('Degenerate Partitions',count3)
      #### End Debugging

    else:
      pass
      # super(PersistenceChartView, self.mousePressEvent(event))

  def contextMenuEvent(self,event):
    self.rightClickMenu.popup(event.globalPos())

  def resizeEvent(self,event):
    super(PersistenceChartView, self).resizeEvent(event)
    self.gView.scale(self.gView.width()/self.scene.width(),
                     self.gView.height()/self.scene.height())
    self.gView.fitInView(self.scene.sceneRect(),qtc.Qt.KeepAspectRatio)
    self.updateScene()

  def update(self):
    super(PersistenceChartView,self).update()

  def persistenceChanged(self):
    self.updateScene()

  def updateScene(self):
    self.scene.clear()

    ## This view will do nothing if it is not given an AMSC object
    if not isinstance(self.segmentation,AMSC_Object):
      return

    if self.fillAction.isChecked():
      self.scene.setSceneRect(0,0,100*float(self.gView.width())/float(self.gView.height()),100)
    else:
      self.scene.setSceneRect(0,0,100,100)

    width = self.scene.width()
    height = self.scene.height()
    minDim = min([width,height])

    self.scene.addRect(0,0,width,height,qtg.QPen(qtc.Qt.black))

    persistence = self.segmentation.Level()
    mergeSequence = self.segmentation.GetMergeSequence()

    minIndices = set(self.segmentation.minIdxs)
    maxIndices = set(self.segmentation.maxIdxs)

    minCount = len(minIndices)
    maxCount = len(maxIndices)
    extCount = minCount+maxCount

    xBounds = [0,max(self.segmentation.Y)-min(self.segmentation.Y)]
    yBounds = [0,minCount+maxCount]
    def scaleToScene(wx,wy):
      tx = float(wx-xBounds[0])/float(xBounds[1]-xBounds[0])
      ty = float(wy-yBounds[0])/float(yBounds[1]-yBounds[0])

      x = tx*width
      y = height - ty*height
      return (x,y)

    persistenceMap = [(p,idx) for idx,(parent,p) in mergeSequence.iteritems()]
    persistenceMap = sorted(persistenceMap)

    minLine = [(0.,minCount)]
    maxLine = [(0.,maxCount)]
    for p,idx in persistenceMap:
      minLine.append((p,minCount))
      maxLine.append((p,maxCount))
      if idx in minIndices:
        minCount -= 1
      else:
        maxCount -= 1
      maxLine.append((p,maxCount))
      minLine.append((p,minCount))

    minLine.append((xBounds[1],minCount))
    maxLine.append((xBounds[1],maxCount))

    gray = qtg.QColor('#999999')
    transparentGray = gray.lighter()
    transparentGray.setAlpha(127)

    persPen = qtg.QPen(gray)
    persBrush = qtg.QBrush(transparentGray)
    (px,py) = scaleToScene(persistence,0)
    (pxHi,py) = scaleToScene(self.segmentation.PersistenceStar(),0)
    # self.scene.addRect(px,0,px,height,persPen)
    self.scene.addRect(px,0,pxHi-px,height,persPen,persBrush)

    if self.extGroup.checkedAction().text() == 'Extrema':
      maxPen = qtg.QPen(maxPenColor)
      maxBrush = qtg.QBrush(maxBrushColor)
      i = 0
      while i < len(maxLine)-1:
        (x1,y1) = maxLine[i]
        (x2,y2) = maxLine[i+1]
        (x1,y1) = scaleToScene(x1,y1+minLine[i][1])
        (x2,y2) = scaleToScene(x2,minLine[i][1])
        if(self.segmentation.Level() <= maxLine[i+1][0]):
          rectItem = self.scene.addRect(x1,y1,(x2-x1),(y2-y1),maxPen,maxBrush)
          rectItem.setToolTip('# Maxima: ' + str(maxLine[i+1][1]) + '\n# Minima:' + str(minLine[i+1][1]))
        else:
          self.scene.addRect(x1,y1,(x2-x1),(y2-y1),maxPen)
        i += 2
      minPen = qtg.QPen(minPenColor)
      minBrush = qtg.QBrush(minBrushColor)
      i = 0
      while i < len(minLine)-1:
        (x1,y1) = minLine[i]
        (x2,y2) = minLine[i+1]
        (x1,y1) = scaleToScene(x1,y1)
        (x2,y2) = scaleToScene(x2,0)
        if(self.segmentation.Level() <= minLine[i+1][0]):
          rectItem = self.scene.addRect(x1,y1,(x2-x1),(y2-y1),minPen,minBrush)
          rectItem.setToolTip('# Maxima: ' + str(maxLine[i+1][1]) + '\n# Minima:' + str(minLine[i+1][1]))
        else:
          self.scene.addRect(x1,y1,(x2-x1),(y2-y1),minPen)
        i += 2

    if self.extGroup.checkedAction().text() == 'Minima':
      minPen = qtg.QPen(minPenColor)
      minBrush = qtg.QBrush(minBrushColor)
      i = 0
      while i < len(minLine)-1:
        (x1,y1) = minLine[i]
        (x2,y2) = minLine[i+1]
        (x1,y1) = scaleToScene(x1,y1)
        (x2,y2) = scaleToScene(x2,0)
        if(self.segmentation.Level() <= minLine[i+1][0]):
          rectItem = self.scene.addRect(x1,y1,(x2-x1),(y2-y1),minPen,minBrush)
          rectItem.setToolTip('Minima count: ' + str(minLine[i+1][1]))
        else:
          self.scene.addRect(x1,y1,(x2-x1),(y2-y1),minPen)
        i += 2

    if self.extGroup.checkedAction().text() == 'Maxima':
      maxPen = qtg.QPen(maxPenColor)
      maxBrush = qtg.QBrush(maxBrushColor)
      i = 0
      while i < len(maxLine)-1:
        (x1,y1) = maxLine[i]
        (x2,y2) = maxLine[i+1]
        (x1,y1) = scaleToScene(x1,y1)
        (x2,y2) = scaleToScene(x2,0)
        if(self.segmentation.Level() <= maxLine[i+1][0]):
          rectItem = self.scene.addRect(x1,y1,(x2-x1),(y2-y1),maxPen,maxBrush)
          rectItem.setToolTip('Maxima count: ' + str(maxLine[i+1][1]))
        else:
          self.scene.addRect(x1,y1,(x2-x1),(y2-y1),maxPen)

        i += 2

    self.gView.fitInView(self.scene.sceneRect(),qtc.Qt.KeepAspectRatio)
