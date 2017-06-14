#!/usr/bin/env python

from qtpy import QtCore as qtc
from qtpy import QtGui as qtg
from qtpy import QtWidgets as qtw
from qtpy import QtSvg as qts

from .BaseTopologicalView import BaseTopologicalView
from AMSC_Object import AMSC_Object

import numpy as np
import operator

from . import colors

class BarcodeGraphicsView(qtw.QGraphicsView):
  def mousePressEvent(self, event):
    super(BarcodeGraphicsView, self).mousePressEvent(event)
    self.parent().mousePressEvent(event,True)

  def mouseMoveEvent(self, event):
    if event.buttons() != qtc.Qt.MiddleButton:
      super(BarcodeGraphicsView, self).mouseMoveEvent(event)
    else:
      self.parent().mouseMoveEvent(event,True)

  def mouseReleaseEvent(self, event):
    if event.button() != qtc.Qt.LeftButton:
      super(BarcodeGraphicsView, self).mouseReleaseEvent(event)
    else:
      self.parent().mouseReleaseEvent(event,True)


class BarcodeView(BaseTopologicalView):
  def __init__(self, parent=None, amsc=None, title="Persistence Barcode"):
    super(BarcodeView, self).__init__(parent,amsc,title)

  def Reinitialize(self, parent=None, amsc=None, title=None):
    # Try to apply a new layout, if one already exists then make sure to grab
    # it for updating
    self.setLayout(qtw.QVBoxLayout())
    layout = self.layout()
    self.clearLayout(layout)

    self.scene = qtw.QGraphicsScene()
    ## TODO: Maybe this one should fill the entire viewing space?
    self.scene.setSceneRect(0,0,100,100)
#    self.scene.setSceneRect(0,0,200,100)
    self.gView = BarcodeGraphicsView(self.scene)
    self.gView.setRenderHints(qtg.QPainter.Antialiasing |
                              qtg.QPainter.SmoothPixmapTransform)
    self.gView.setHorizontalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)
    self.gView.setVerticalScrollBarPolicy(qtc.Qt.ScrollBarAlwaysOff)

    self.rightClickMenu = qtw.QMenu()

    blankAction = self.rightClickMenu.addAction('Force Screen Update')
    blankAction.triggered.connect(self.blank)
    self.sortMenu = self.rightClickMenu.addMenu('Sort By')
    sortActions = []
    sortActions.append(self.sortMenu.addAction('Persistence'))
    sortActions.append(self.sortMenu.addAction('Birth'))
    sortActions.append(self.sortMenu.addAction('Death'))
    sortActions.append(self.sortMenu.addAction('Size'))
    self.sortGroup = qtw.QActionGroup(self.sortMenu)
    for act in sortActions:
      act.setCheckable(True)
      self.sortGroup.addAction(act)
    sortActions[1].setChecked(True)
    self.sortGroup.triggered.connect(self.updateScene)

    self.widthMenu = self.rightClickMenu.addMenu('Width')
    widthActions = []
    widthActions.append(self.widthMenu.addAction('Equal'))
    widthActions.append(self.widthMenu.addAction('Fixed'))
    widthActions.append(self.widthMenu.addAction('Variable'))
    self.widthGroup = qtw.QActionGroup(self.widthMenu)
    for act in widthActions:
      act.setCheckable(True)
      self.widthGroup.addAction(act)
    widthActions[0].setChecked(True)
    self.widthGroup.triggered.connect(self.updateScene)

    self.fillAction = self.rightClickMenu.addAction('Fill viewport')
    self.fillAction.setCheckable(True)
    self.fillAction.setChecked(True)
    self.fillAction.triggered.connect(self.updateScene)

    incAction = self.rightClickMenu.addAction('Increase Persistence')
    incAction.triggered.connect(self.increasePersistence)
    decAction = self.rightClickMenu.addAction('Decrease Persistence')
    decAction.triggered.connect(self.decreasePersistence)

    captureAction = self.rightClickMenu.addAction('Capture')
    captureAction.triggered.connect(self.saveImage)

    self.gView.scale(self.gView.width()/self.scene.width(),
                     self.gView.height()/self.scene.height())

    layout.addWidget(self.gView)
    self.updateScene()

  def increasePersistence(self):
    """ A method for increasing the persistence to the next coarsest level given
        the current setting.
    """
    mergeSequence = self.amsc.GetMergeSequence()
    persistences = [p for idx,(parent,p) in mergeSequence.items()]
    eps = max(persistences)*1e-6
    persistences = sorted(set(persistences))
    persistences.insert(0,0.)
    persistences.pop()

    currentP = self.amsc.Persistence()
    idx = 0
    while persistences[idx]+eps <= currentP and idx < len(persistences)-1:
      idx += 1
    self.amsc.Persistence(persistences[idx]+eps)

  def decreasePersistence(self):
    """ A method for decreasing the persistence to the next finest level given
        the current setting.
    """
    mergeSequence = self.amsc.GetMergeSequence()
    persistences = [p for idx,(parent,p) in mergeSequence.items()]
    eps = max(persistences)*1e-6
    persistences = sorted(set(persistences))
    persistences.insert(0,0.)
    persistences.pop()

    currentP = self.amsc.Persistence()
    idx = len(persistences)-1
    while persistences[idx]+eps >= currentP and idx > 0:
      idx -= 1
    self.amsc.Persistence(persistences[idx]+eps)

  def blank(self):
    self.updateScene()

  def resizeEvent(self,event):
    super(BarcodeView, self).resizeEvent(event)
    self.gView.scale(self.gView.width()/self.scene.width(),
                     self.gView.height()/self.scene.height())
    self.gView.fitInView(self.scene.sceneRect(),qtc.Qt.KeepAspectRatio)
    self.updateScene()

  def saveImage(self):
    ## This view will do nothing if it is not given an AMSC object
    if not isinstance(self.amsc,AMSC_Object):
      return
    dialog = qtw.QFileDialog(self)
    dialog.setFileMode(qtw.QFileDialog.AnyFile)
    dialog.setAcceptMode(qtw.QFileDialog.AcceptSave)
    dialog.exec_()
    if dialog.result() == qtw.QFileDialog.Accepted:
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
        image.fill(qtc.Qt.black)
        painter = qtg.QPainter(image)
      self.scene.render(painter)
      if myFile.endswith('.svg'):
        pass
      else:
        image.save(myFile,quality=100)
      del painter

  def mousePressEvent(self, event, fromChild=False):
    if fromChild:
      mousePt = self.gView.mapToScene(event.x(),event.y())
    else:
      pass
      # super(PersistenceView, self.mousePressEvent(event))

  def mouseMoveEvent(self, event, fromChild=False):
    if fromChild:
      pass
    else:
      pass
      # super(PersistenceView, self.mousePressEvent(event))

  def mouseReleaseEvent(self, event, fromChild=False):
    ## This view will do nothing if it is not given an AMSC object
    if not isinstance(self.amsc,AMSC_Object):
      return
    if fromChild:
      mergeSequence = self.amsc.GetMergeSequence()
      mousePt = self.gView.mapToScene(event.x(),event.y()).y()
      persistences = [p for idx,(parent,p) in mergeSequence.items()]
      persistences = sorted(set(persistences))
      minP = 0
      maxP = max(persistences)
      width = self.scene.width()
      persistence = np.clip(mousePt/width*(maxP-minP) + minP, minP, maxP)
      # print(mousePt, persistence)
      # idx = len(persistences)-1
      # while persistences[idx] >= persistence and idx > 0:
      #   idx -= 1
      # self.amsc.Persistence(persistences[idx])
      # self.amsc.Persistence(persistence)
    else:
      pass
      # super(PersistenceView, self.mousePressEvent(event))

  def sizeHint(self):
    return qtc.QSize(300,600)

  def contextMenuEvent(self,event):
    self.rightClickMenu.popup(event.globalPos())

  def update(self):
    super(BarcodeView,self).update()

  def persistenceChanged(self):
    self.updateScene()

  def selectionChanged(self):
    self.updateScene()

  def updateScene(self):
    self.scene.clear()

    if self.fillAction.isChecked():
      self.scene.setSceneRect(0,0,100*float(self.gView.width())/float(self.gView.height()),100)
    else:
      self.scene.setSceneRect(0,0,100,100)

    width = self.scene.width()
    height = self.scene.height()
    minDim = min([width,height])

    mergeSequence = self.amsc.GetMergeSequence()
    Ys = self.amsc.GetY(None)

    minY = min(Ys)
    maxY = max(Ys)
    currentP = self.amsc.Persistence()
    partitions = self.amsc.Partitions(currentP)
    bars  = []

    counts = {}
    allCount = 0
    for key in mergeSequence.keys():
      counts[key] = 0
    for extPair,items in partitions.items():
      counts[extPair[0]] += len(items)
      counts[extPair[1]] += len(items)

    allCount = sum(counts.values())

    # We would like to sort the persistence barcode by the birth times, so we
    # need to define a little local function to give us that value to pass to
    # the sort function
    reverseSort = False
    if self.sortGroup.checkedAction().text() == 'Birth':
      def sortIndex(x):
        if self.amsc.GetClassification(x[0]) == 'minimum':
          return Ys[x[0]]
        else:
          return Ys[x[0]]-x[1][1]
    elif self.sortGroup.checkedAction().text() == 'Death':
      reverseSort = True
      def sortIndex(x):
        if self.amsc.GetClassification(x[0]) == 'maximum':
          return Ys[x[0]]
        else:
          return Ys[x[0]]+x[1][1]
    elif self.sortGroup.checkedAction().text() == 'Size':
      reverseSort = True
      def sortIndex(x):
          return counts[x[0]]
    else:
      reverseSort = True
      def sortIndex(x):
        return x[1][1]

    sortedSequence = sorted(mergeSequence.items(), key=sortIndex, reverse=reverseSort)

    numInactive = 0
    for idx,(parent,p) in sortedSequence:
      if p < currentP:
        numInactive += 1
    numActive = len(sortedSequence)-numInactive

    # The inactive should take at most a quarter of the screen
    if self.widthGroup.checkedAction().text() == 'Equal':
      usableHeight = height
      activeHeight = inactiveHeight = height/float(numActive+numInactive)
    else:
      if numInactive == 0:
        inactiveHeight = 3.
      else:
        inactiveHeight = min(3.,height*0.25/float(numInactive))

      usableHeight = height-inactiveHeight*numInactive
      activeHeight = usableHeight/numActive

    currentY = 0
    for idx,(parent,p) in sortedSequence:
      myY = Ys[idx]
      barWidth = p/(maxY-minY)*width
      if p >= currentP and self.amsc.GetClassification(idx) == 'minimum':
        fillColor = colors.minBrushColor
        penColor = colors.minPenColor
      elif self.amsc.GetClassification(idx) == 'minimum':
        fillColor = colors.inactiveMinBrushColor
        penColor = colors.inactiveMinPenColor
      elif p >= currentP:
        fillColor = colors.maxBrushColor
        penColor = colors.maxPenColor
      else:
        fillColor = colors.inactiveMaxBrushColor
        penColor = colors.inactiveMaxPenColor

      if self.widthGroup.checkedAction().text() == 'Variable':
        # Normalize the count of this guy by the count of everything that needs
        # to be accounted for.
        if p >= currentP:
          barHeight = float(counts[idx])/allCount*(usableHeight)
        else:
          barHeight = inactiveHeight
      elif p >= currentP:
        barHeight = activeHeight
      else:
        barHeight = inactiveHeight

      if self.sortGroup.checkedAction().text() in ['Birth','Death']:
        if self.amsc.GetClassification(idx) == 'minimum':
          barStartX = (myY-minY)/(maxY-minY)*width
        else:
          barStartX = (myY-p-minY)/(maxY-minY)*width
      else:
        barStartX = 0
      if idx in self.amsc.selectedExtrema:
        myPen = qtg.QPen(qtc.Qt.black)
      else:
        myPen = qtg.QPen(penColor)

      myPen.setCosmetic(True)
      # fillColor.setAlphaF(0.5)
      # self.scene.addEllipse(barStartX,currentY,barWidth,barHeight,myPen,qtg.QBrush(barColor))
      myRect = self.scene.addRect(barStartX,currentY,barWidth,barHeight,myPen,qtg.QBrush(fillColor))
      if idx == parent:
        myRect.setToolTip(str(idx) + ' (' + str(p) + ')')
      else:
        myRect.setToolTip(str(idx) + ' (' + str(p) + ') -> ' + str(parent))
      currentY += barHeight
    outlinePen = qtg.QPen(qtc.Qt.black)
    outlinePen.setCosmetic(True)
    self.scene.addRect(0,0,width,height,outlinePen)
    self.gView.fitInView(self.scene.sceneRect(),qtc.Qt.KeepAspectRatio)
