#!/usr/bin/env python

from PySide import QtCore as qtc
from PySide import QtGui as qtg

from GenericView import GenericView
from AMSC_Object import AMSC_Object

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm, colors, font_manager

import numpy as np
import math

from colors import *

C = np.array([[0,90,151],[0,90,151],[0,90,151],[1,90,151],[1,91,151],[1,91,151],
              [1,91,151],[1,91,151],[1,91,151],[2,92,151],[2,92,151],[2,92,151],
              [2,92,150],[2,92,150],[2,93,150],[3,93,150],[3,93,150],[3,93,150],
              [3,94,150],[3,94,150],[3,94,150],[4,94,150],[4,94,150],[4,95,150],
              [4,95,149],[4,95,149],[4,95,149],[5,96,149],[5,96,149],[5,96,149],
              [5,96,149],[5,96,149],[5,97,149],[6,97,149],[6,97,149],[6,97,148],
              [6,97,148],[6,98,148],[6,98,148],[7,98,148],[7,98,148],[7,99,148],
              [7,99,148],[7,99,148],[7,99,148],[8,99,148],[8,100,148],
              [8,100,147],[8,100,147],[8,100,147],[8,101,147],[9,101,147],
              [9,101,147],[9,101,147],[9,101,147],[9,102,147],[10,102,147],
              [10,102,147],[10,102,147],[10,103,146],[10,103,146],[10,103,146],
              [11,103,146],[11,103,146],[11,104,146],[12,105,145],[13,106,144],
              [13,107,142],[14,108,141],[15,109,140],[16,110,139],[16,111,138],
              [17,112,137],[18,113,136],[19,114,134],[20,116,133],[20,117,132],
              [21,118,131],[22,119,130],[23,120,129],[23,121,128],[24,122,127],
              [25,123,125],[26,124,124],[27,125,123],[27,126,122],[28,127,121],
              [29,128,120],[30,129,119],[30,130,118],[31,132,116],[32,133,115],
              [33,134,114],[34,135,113],[34,136,112],[35,137,111],[36,138,110],
              [37,139,109],[37,140,107],[38,141,106],[39,142,105],[40,143,104],
              [40,144,103],[41,145,102],[42,146,101],[43,148,100],[44,149,98],
              [44,150,97],[45,151,96],[46,152,95],[47,153,94],[47,154,93],
              [48,155,92],[49,156,91],[50,157,89],[51,158,88],[51,159,87],
              [52,160,86],[53,161,85],[54,162,84],[54,164,83],[55,165,82],
              [56,166,80],[57,167,79],[58,168,78],[58,169,77],[59,170,76],
              [60,171,75],[61,171,74],[64,171,73],[66,171,72],[69,171,71],
              [71,171,70],[74,170,70],[76,170,69],[79,170,68],[81,170,67],
              [84,169,66],[86,169,65],[88,169,64],[91,169,64],[93,169,63],
              [96,168,62],[98,168,61],[101,168,60],[103,168,59],[106,168,58],
              [108,167,58],[110,167,57],[113,167,56],[115,167,55],[118,166,54],
              [120,166,53],[123,166,53],[125,166,52],[128,166,51],[130,165,50],
              [133,165,49],[135,165,48],[137,165,47],[140,164,47],[142,164,46],
              [145,164,45],[147,164,44],[150,164,43],[152,163,42],[155,163,42],
              [157,163,41],[159,163,40],[162,162,39],[164,162,38],[167,162,37],
              [169,162,36],[172,162,36],[174,161,35],[177,161,34],[179,161,33],
              [181,161,32],[184,161,31],[186,160,30],[189,160,30],[191,160,29],
              [194,160,28],[196,159,27],[199,159,26],[201,159,25],[204,159,25],
              [206,159,24],[208,158,23],[211,158,22],[213,158,21],[216,158,20],
              [217,158,20],[217,158,20],[218,157,19],[219,157,19],[219,157,19],
              [220,157,18],[221,157,18],[221,157,18],[222,157,17],[222,157,17],
              [223,157,17],[224,157,16],[224,157,16],[225,157,16],[225,157,15],
              [226,156,15],[227,156,15],[227,156,14],[228,156,14],[229,156,14],
              [229,156,14],[230,156,13],[230,156,13],[231,156,13],[232,156,12],
              [232,156,12],[233,156,12],[234,156,11],[234,156,11],[235,155,11],
              [235,155,10],[236,155,10],[237,155,10],[237,155,9],[238,155,9],
              [238,155,9],[239,155,9],[240,155,8],[240,155,8],[241,155,8],
              [242,155,7],[242,155,7],[243,155,7],[243,154,6],[244,154,6],
              [245,154,6],[245,154,5],[246,154,5],[247,154,5],[247,154,4],
              [248,154,4],[248,154,4],[249,154,3],[250,154,3],[250,154,3],
              [251,154,3],[251,154,2],[252,153,2],[253,153,2],[253,153,1],
              [254,153,1],[255,153,1],[255,153,0],[255,153,0]],dtype=float)
C /= 255.0
myColormap = colors.ListedColormap(C)

class PersistenceDiagramView(GenericView):
  def __init__(self, parent=None, amsc=None, title="Persistence Diagram"):
    super(PersistenceDiagramView, self).__init__(parent,amsc,title)

  def Reinitialize(self, parent=None, amsc=None, title="Persistence Diagram"):
    # Try to apply a new layout, if one already exists then make sure to grab
    # it for updating
    self.setLayout(qtg.QVBoxLayout())
    layout = self.layout()
    self.clearLayout(layout)

    self.fig = Figure(facecolor='white')
    self.mplCanvas = FigureCanvas(self.fig)
    self.mplCanvas.axes = self.fig.add_subplot(111)
    # We want the axes cleared every time plot() is called
    self.mplCanvas.axes.hold(False)
    self.colorbar = None

    self.rightClickMenu = qtg.QMenu()
    self.colorAction = self.rightClickMenu.addAction('Min/Max Colors')
    self.colorAction.setCheckable(True)
    self.colorAction.setChecked(True)
    self.colorAction.triggered.connect(self.updateScene)

    layout.addWidget(self.mplCanvas)
    self.updateScene()

  def sizeHint(self):
    return qtc.QSize(300,600)

  def update(self):
    super(PersistenceDiagramView,self).update()

  def persistenceChanged(self):
    self.updateScene()

  def selectionChanged(self):
    pass

  def updateScene(self):
    ## This view will do nothing if it is not given an AMSC object
    if not isinstance(self.segmentation,AMSC_Object):
      return

    fontSize=24
    smallFontSize=20
    self.mplCanvas.axes.clear()
    mergeSequence = self.segmentation.GetMergeSequence()
    Ys = self.segmentation.GetY(None)
    currentP = self.segmentation.Level()
    minY = min(Ys)
    maxY = max(Ys)
    minP = 0
    maxP = 0
    minActiveBirthVals = []
    minActiveDeathVals = []

    minInactiveBirthVals = []
    minInactiveDeathVals = []

    maxActiveBirthVals = []
    maxActiveDeathVals = []

    maxInactiveBirthVals = []
    maxInactiveDeathVals = []

    minActivePVals = []
    maxActivePVals = []

    minInactivePVals = []
    maxInactivePVals = []

    activeSize = 80
    inactiveSize = 40

    for idx,(parent,p) in mergeSequence.iteritems():
      myY = Ys[idx]
      maxP = max(maxP,p)
      if self.segmentation.GetClassification(idx) == 'minimum':
        if p >= currentP:
          minActiveBirthVals.append(myY)
          minActiveDeathVals.append(myY+p)
          minActivePVals.append(p)
        else:
          minInactiveBirthVals.append(myY)
          minInactiveDeathVals.append(myY+p)
          minInactivePVals.append(p)
      else:
        if p >= currentP:
          maxActiveBirthVals.append(myY-p)
          maxActiveDeathVals.append(myY)
          maxActivePVals.append(p)
        else:
          maxInactiveBirthVals.append(myY-p)
          maxInactiveDeathVals.append(myY)
          maxInactivePVals.append(p)

    if self.colorAction.isChecked():
      myPlot = self.mplCanvas.axes.scatter(minActiveBirthVals,
                                           minActiveDeathVals,
                                           c=minBrushColor.name(),
                                           edgecolors=minPenColor.name(),
                                           marker='v',s=activeSize)
      self.mplCanvas.axes.hold(True)
      myPlot = self.mplCanvas.axes.scatter(minInactiveBirthVals,
                                           minInactiveDeathVals,
                                           c=inactiveMinBrushColor.name(),
                                           edgecolors=inactiveMinPenColor.name(),
                                           marker='v',s=inactiveSize)
      myPlot = self.mplCanvas.axes.scatter(maxActiveBirthVals,
                                           maxActiveDeathVals, c=maxBrushColor.name(),
                                           edgecolors=maxPenColor.name(),
                                           marker='^',s=activeSize)
      myPlot = self.mplCanvas.axes.scatter(maxInactiveBirthVals,
                                           maxInactiveDeathVals,
                                           c=inactiveMaxBrushColor.name(),
                                           edgecolors=inactiveMaxPenColor.name(),
                                           marker='^',s=inactiveSize)
    else:
      myPlot = self.mplCanvas.axes.scatter(minActiveBirthVals,
                                           minActiveDeathVals,c=minActivePVals,
                                           cmap=myColormap,vmin=minP,vmax=maxP,
                                           edgecolors='none',marker='v',
                                           s=activeSize)
      self.mplCanvas.axes.hold(True)
      myPlot = self.mplCanvas.axes.scatter(minInactiveBirthVals,
                                           minInactiveDeathVals,
                                           c=minInactivePVals,cmap=myColormap,
                                           vmin=minP,vmax=maxP,
                                           edgecolors='none',marker='v',
                                           s=inactiveSize)
      myPlot = self.mplCanvas.axes.scatter(maxActiveBirthVals,
                                           maxActiveDeathVals,c=maxActivePVals,
                                           cmap=myColormap,vmin=minP,vmax=maxP,
                                           edgecolors='none',marker='^',
                                           s=activeSize)
      myPlot = self.mplCanvas.axes.scatter(maxInactiveBirthVals,
                                           maxInactiveDeathVals,
                                           c=maxInactivePVals,cmap=myColormap,
                                           vmin=minP,vmax=maxP,
                                           edgecolors='none',marker='^',
                                           s=inactiveSize)
      self.colorbar = self.fig.colorbar(myPlot)
    # minVal = 3e7
    # maxVal = 7.5e7
    # ticks = np.linspace(0, 1.8, 4)
    minVal = min(self.segmentation.GetY())
    maxVal = max(self.segmentation.GetY())
    self.mplCanvas.axes.plot([minVal,maxVal],[minVal,maxVal],c=myColormap(minP))
    self.mplCanvas.axes.plot([minVal,maxVal],[minVal+currentP,maxVal+currentP],
                             'k--')
    ticks = np.linspace(minVal,maxVal,2)
    self.mplCanvas.axes.set_xlim(ticks[0],ticks[-1])
    self.mplCanvas.axes.set_ylim(ticks[0],ticks[-1])
    self.mplCanvas.axes.set_yticks([])
    self.mplCanvas.axes.set_xticks([])

    self.mplCanvas.axes.set_xlabel('Birth',size=fontSize)
    self.mplCanvas.axes.set_ylabel('Death',size=fontSize)

    for label in  (self.mplCanvas.axes.get_xticklabels()+self.mplCanvas.axes.get_yticklabels()):
      label.set_fontsize(smallFontSize)
    # self.mplCanvas.axes.set_xlim(left=0)
    # self.mplCanvas.axes.set_ylim(bottom=0)

    self.mplCanvas.axes.hold(False)
    self.mplCanvas.draw()

