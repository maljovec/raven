from PySide import QtCore as qtc
from PySide import QtGui as qtg

import sys
import os

myPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(myPath,'contrib','AMSC'))

# We will not support external running of this UI, it shall be run from RAVEN
if __name__ == '__main__':
  app = qtg.QApplication(sys.argv)

  X = None
  Y = None
  if len(sys.argv) > 1:
    print('\tYou probably want me to load a file...')
    print('\tThe Maker has not included this in my programming.')

  from UI import TopologyWindow

  main = TopologyWindow(X,Y)
  main.show()
  sys.exit(app.exec_())