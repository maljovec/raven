from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QGridLayout, QHBoxLayout, QVBoxLayout
from qtpy.QtWidgets import QDialogButtonBox, QGroupBox
from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QSpinBox, QDoubleSpinBox

class DataInterpreterDialog(QDialog):
  def __init__(self,parent=None,dimNames=None,data=None):
    super(DataInterpreterDialog,self).__init__(parent=parent)
    self.names = dimNames
    self.data = data
    layout = QGridLayout()

    row = 0
    col = 0
    self.inputChks = []
    inputGrpBox = QGroupBox('Inputs')
    subLayout = QVBoxLayout()
    for dim in self.names:
      self.inputChks.append(QCheckBox(dim))
      subLayout.addWidget(self.inputChks[-1])
      self.inputChks[-1].setChecked(True)
    self.inputChks[-1].setChecked(False)
    inputGrpBox.setLayout(subLayout)
    layout.addWidget(inputGrpBox,row,col,1,2)

    row += 1
    col = 0
    layout.addWidget(QLabel('Output:'),row,col)
    col += 1
    self.outputCmbBox = QComboBox()
    self.outputCmbBox.addItems(self.names)
    layout.addWidget(self.outputCmbBox,row,col)

    # row += 1
    # col = 0
    # layout.addWidget(QLabel('Weights:'),row,col)
    # col += 1
    # weightsGroupBox = QGroupBox()
    # subLayout = QHBoxLayout()
    # self.weightsChkBox = QCheckBox('From Input File')
    # subLayout.addWidget(self.weightsChkBox)
    # self.weightsCmbBox = QComboBox()
    # self.weightsCmbBox.addItems(self.names)
    # subLayout.addWidget(self.weightsCmbBox)
    # weightsGroupBox.setLayout(subLayout)
    # layout.addWidget(weightsGroupBox,row,col)
    # self.weightsChkBox.stateChanged.connect(self.weightToggled)

    row += 1
    col = 0
    layout.addWidget(QLabel('Normalization:'),row,col)
    col += 1
    self.normCmbBox = QComboBox()
    self.normCmbBox.addItem('None')
    self.normCmbBox.addItem('Range Scaling')
    self.normCmbBox.addItem('Z-Score Standardization')
    self.normCmbBox.setCurrentIndex(2)
    layout.addWidget(self.normCmbBox,row,col)

    row += 1
    col = 0
    layout.addWidget(QLabel('Graph:'),row,col)
    col += 1
    self.graphCmbBox = QComboBox()
    self.graphCmbBox.addItem('Approximate KNN')
    self.graphCmbBox.addItem('Beta Skeleton')
    self.graphCmbBox.addItem('Relaxed Beta Skeleton')
    layout.addWidget(self.graphCmbBox,row,col)
    self.graphCmbBox.currentIndexChanged.connect(self.graphChanged)

    row += 1
    col = 0
    layout.addWidget(QLabel('Gradient:'),row,col)
    col += 1
    self.gradCmbBox = QComboBox()
    self.gradCmbBox.addItem('Steepest')
    layout.addWidget(self.gradCmbBox,row,col)

    row += 1
    col = 0
    layout.addWidget(QLabel('k:'),row,col)
    col += 1
    self.kSpnBox = QSpinBox()
    self.kSpnBox.setMinimum(2)
    self.kSpnBox.setMaximum(len(self.data)-1)
    # self.kSpnBox.setValue(len(self.data)-1)
    self.kSpnBox.setValue(50)
    layout.addWidget(self.kSpnBox,row,col)

    row += 1
    col = 0
    layout.addWidget(QLabel('beta:'),row,col)
    col += 1
    self.betaSpnBox = QDoubleSpinBox()
    self.betaSpnBox.setValue(1)
    self.betaSpnBox.setMinimum(0.1)
    self.betaSpnBox.setMaximum(2)
    self.betaSpnBox.setSingleStep(0.1)
    layout.addWidget(self.betaSpnBox,row,col)

    row += 1
    col = 0
    buttons = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel,
                               Qt.Horizontal, self)
    layout.addWidget(buttons,row,1)
    buttons.accepted.connect(self.accept)
    buttons.rejected.connect(self.reject)
    self.setLayout(layout)

    self.graphCmbBox.setCurrentIndex(2)
    self.outputCmbBox.setCurrentIndex(self.outputCmbBox.count()-1)
    # self.weightsChkBox.setChecked(True)
    # self.weightsChkBox.setChecked(False)

  def graphChanged(self):
    self.betaSpnBox.setEnabled(self.graphCmbBox.currentText().endswith('Beta Skeleton'))

  # def weightToggled(self):
  #   self.weightsCmbBox.setEnabled(self.weightsChkBox.isChecked())

  def accept(self):
    self.inputColumns = []
    for i,item in enumerate(self.inputChks):
      if item.isChecked():
        self.inputColumns.append(i)
    self.outputColumn = self.outputCmbBox.currentIndex()
    # if self.weightsChkBox.isChecked():
    #   self.weights = self.data[:,self.weightsCmbBox.currentIndex()]
    # else:
    #   self.weights = None
    self.weights = None

    if self.normCmbBox.currentText() == 'Feature Scaling':
      self.normalization = 'feature'
    elif self.normCmbBox.currentText() == 'Z-Score Standardization':
      self.normalization = 'zscore'
    else:
      self.normalization = None
    self.graph = str(self.graphCmbBox.currentText().lower())
    self.gradient = str(self.gradCmbBox.currentText().lower())
    self.beta = self.betaSpnBox.value()
    self.k = self.kSpnBox.value()
    super(DataInterpreterDialog,self).accept()