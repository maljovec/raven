from qtpy import QtCore as qtc
from qtpy import QtGui as qtg
from qtpy import QtWidgets as qtw

import sys
import os

myPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(myPath,'contrib','AMSC'))

# We will not support external running of this UI, it shall be run from RAVEN
if __name__ == '__main__':
  app = qtw.QApplication(sys.argv)

  X = None
  Y = None
  # if len(sys.argv) > 1:
  #   print('\tYou probably want me to load a file...')
  #   print('\tThe Maker has not included this in my programming.')

  from UI import TopologyWindow

  main = TopologyWindow(X,Y,debug=True)
  main.loadData()
  main.show()
  main.addNewView('TopologyMapView')
  main.addNewView('ScatterView2D')
  main.addNewView('SensitivityView')
  main.addNewView('FitnessView')
  sys.exit(app.exec_())