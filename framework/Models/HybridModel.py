# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created on September, 2017

@author: wangc
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import copy
import numpy as np
import time
from collections import OrderedDict
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from utils import utils
from .Dummy import Dummy
from utils import InputData
import Runners
#Internal Modules End--------------------------------------------------------------------------------

class HybridModel(Dummy):
  """
    HybridModel Class. This class is aimed to automatically select the model to run among different models
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for the class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying input of cls.
    """
    inputSpecification = super(HybridModel, cls).getInputSpecification()

    # fill in the inputSpecification

    return inputSpecification

  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    pass

  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    Dummy.__init__(self,runInfoDict)
    self.modelInstance            = None
    self.cvInstance               = None
    self.targetEvaluationInstance = None
    self.romsDictionary        = {}      # dictionary of models that is going to be employed, i.e. {'romName':Instance}
    self.romTrainStepSize      = 5        # the step size for rom train
    self.romTrainStartSize     = 10       # the initial size of training set
    self.romTrainMaxSize       = 1000     # the maximum size of training set
    self.romValidateSize       = 10       # the size of rom validation set
    self.counter               = 0        # record the number of model runs
    self.romTrained            = False    # True if all roms are trained
    self.sleepTime             = 0.005    # waiting time before checking if a run is finished.
    self.romConverged          = False    # True if all roms are converged
    self.romValid              = False    # True if all roms are valid for given input data
    self.romConvergence        = 0.001
    self.tempOutputs           = {}
    self.optimizerTypes        = ['SPSA'] # list of types of optimizer
    self.printTag              = 'HYBRIDMODEL MODEL' # print tag
    # assembler objects to be requested
    self.addAssemblerObject('Model','1',True)
    self.addAssemblerObject('ROM','n')
    self.addAssemblerObject('CV','1')
    self.addAssemblerObject('TargetEvaluation','1')

  def localInputAndChecks(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    Dummy.localInputAndChecks(self, xmlNode)
    for child in xmlNode:
      if child.tag == 'settings':
        self.__readSettings(child)
      elif child.tag == 'Model':
        self.modelInstance = child.text.strip()
      elif child.tag == 'CV':
        self.cvInstance = child.text.strip()
      elif child.tag == 'TargetEvaluation':
        self.targetEvaluationInstance = child.text.strip()
      elif child.tag == 'ROM':
        romName = child.text.strip()
        # 'useRom': [converged, validated]
        self.romsDictionary[romName] = {'Instance':None,'Converged':False,'Valid':False}

  def __readSettings(self, xmlNode):
    """
      Method to read the model settings from XML input files
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == 'trainStep':
        self.romTrainStepSize  = int(child.text)
      if child.tag == 'maxTrainSize':
        self.romTrainMaxSize = int(child.text)
      if child.tag == 'initialTrainSize':
        self.romTrainStartSize = 10
      if child.tag == 'tolerance':
        self.romConvergence = float(child.text)

  def initialize(self,runInfo,inputs,initDict=None):
    """
      Method to initialize this model class
      @ In, runInfo, dict, is the run info from the jobHandler
      @ In, inputs, list, is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
      @ Out, None
    """
    self.tempOutputs['uncollectedJobIds'] = []
    self.modelInstance = self.retrieveObjectFromAssemblerDict('Model', self.modelInstance)
    self.cvInstance = self.retrieveObjectFromAssemblerDict('CV', self.cvInstance)
    self.targetEvaluationInstance = self.retrieveObjectFromAssemblerDict('TargetEvaluation', self.targetEvaluationInstance)
    if self.modelInstance is None:
      self.raiseAnError(IOError,'Model XML block needs to be inputted!')
    if self.cvInstance is None:
      self.raiseAnError(IOError, 'CV XML block needs to be inputted!')
    if self.targetEvaluationInstance is None:
      self.raiseAnError(IOError, 'TargetEvaluation XML block needs to be inputted!')

    for romName, romInfo in self.romsDictionary.items():
      romInfo['Instance'] = self.retrieveObjectFromAssemblerDict('ROM', romName)
      if romInfo['Instance']  is None:
        self.raiseAnError(IOError, 'ROM XML block needs to be inputted!')

    modelInputs = self.targetEvaluationInstance.getParaKeys("inputs")
    modelOutputs = self.targetEvaluationInstance.getParaKeys("outputs")
    modelName = self.modelInstance.name
    totalRomOutputs = []

    for romInfo in self.romsDictionary.values():
      romIn = romInfo['Instance']
      romInputs = romIn.getInitParams()['Features']
      romOutputs = romIn.getInitParams()['Target']
      totalRomOutputs.extend(romOutputs)
      unknownList = utils.checkIfUnknowElementsinList(modelInputs, romInputs)
      if unknownList:
        self.raiseAnError(IOError, 'Input Parameters: "', ','.join(str(e) for e in unknownList), '" used in ROM ', romIn.name, ' can not found in Model ', modelName)
      unknownList = utils.checkIfUnknowElementsinList(romInputs, modelInputs)
      if unknownList:
        self.raiseAnError(IOError, 'Input Parameters: "', ','.join(str(e) for e in unknownList), '" used in Model ', modelName, ', but not used in ROM ', romIn.name)
      unknownList = utils.checkIfUnknowElementsinList(modelOutputs, romOutputs)
      if unknownList:
        self.raiseAnError(IOError, 'Output Parameters: "', ','.join(str(e) for e in unknownList), '" used in ROM ', romIn.name, ' can not found in Model ', modelName)
      if romIn.amITrained:
        # Only untrained roms are allowed
        self.raiseAnError(IOError,'HybridModel only accepts untrained ROM, but rom "', romIn.name, '" is already trained')

    # check: we require that the union of ROMs outputs is the same as the paired model in order to use the ROM
    # to replace the paired model.
    if len(set(totalRomOutputs)) != len(totalRomOutputs):
      dup = []
      for elem in set(totalRomOutputs):
        if totalRomOutputs.count(elem) > 1:
          dup.append(elem)
      # we assume there is no duplicate outputs among the roms
      self.raiseAnError(IOError, 'The following outputs ', ','.join(str(e) for e in dup), "are found in the outputs of multiple roms!")
    unknownList = utils.checkIfUnknowElementsinList(totalRomOutputs,modelOutputs)
    if unknownList:
      self.raiseAnError(IOError, "The following outputs: ", ','.join(str(e) for e in unknownList), " used in Model: ", modelName, "but not used in the paired ROMs.")

  def getInitParams(self):
    """
      Method used to export to the printer in the base class the additional PERMANENT your local class have
      @ In, None
      @ Out, tempDict, dict, dictionary to be updated. {'attribute name':value}
    """
    tempDict = OrderedDict()
    tempDict['ROMs contained in HybridModel are '] = self.romsDictionary.keys()
    return tempDict

  def __selectInputSubset(self,romName, kwargs):
    """
      Method aimed to select the input subset for a certain model
      @ In, romName, string, the rom name
      @ In, kwargs , dict, the kwarded dictionary where the sampled vars are stored
      @ Out, selectedKwargs , dict, the subset of variables (in a swallow copy of the kwargs dict)
    """
    selectedKwargs = copy.copy(kwargs)
    selectedKwargs['SampledVars'], selectedKwargs['SampledVarsPb'] = {}, {}
    for key in kwargs["SampledVars"].keys():
      if key in self.romsDictionary[romName]['Instance'].getInitParams()['Features']:
        selectedKwargs['SampledVars'][key] = kwargs["SampledVars"][key]
        selectedKwargs['SampledVarsPb'][key] = kwargs["SampledVarsPb"][key] if 'SampledVarsPb' in kwargs.keys() else 1.0
    return selectedKwargs

  def createNewInput(self,myInput,samplerType,**kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, newInputs, dict, dict that returns the new inputs for each sub-model
    """
    self.raiseADebug("Create New Input")
    if self.romValid:
      identifier = kwargs['prefix']
      newKwargs = {'prefix':identifier}
      for romName in self.romsDictionary.keys():
        newKwargs[romName] = self.__selectInputSubset(romName, kwargs)
        newKwargs[romName]['prefix'] = romName+utils.returnIdSeparator()+identifier
        newKwargs[romName]['uniqueHandler'] = self.name+identifier
    else:
      newKwargs = copy.deepcopy(kwargs)

    return (myInput, samplerType, newKwargs)


  def trainRom(self):
    """
      This function will train all ROMs if they are not converged
      @ In, None
      @ Out, None
    """
    self.raiseADebug("Start to train roms")
    self.raiseADebug("Current sample: ", self.counter)
    for romInfo in self.romsDictionary.values():
      if not romInfo['Converged']:
        romInfo['Instance'].train(self.targetEvaluationInstance)
        converged = self.isRomConverged()
        if converged:
          romInfo['Converged'] = True
          self.raiseADebug("ROM ", romInfo['Instance'].name, " is converged!")

  def checkRomConvergence(self):
    """
      This function will check the convergence of all ROMs
      @ In, None
      @ Out, bool, True if all ROMs are converged
    """
    converged = True
    for romInfo in self.romsDictionary.values():
      if not romInfo['Converged']:
        converged = False
    if converged:
      self.raiseADebug("All ROMs are converged")
    return converged


  def checkRomValidity(self):
    """
      This function will check the validity of all roms
      @ In, None
      @ Out, None
    """
    allValid = True
    for romInfo in self.romsDictionary.values():
      valid = self.isRomValid()
      romInfo['Valid'] = valid
      if valid:
        self.raiseADebug("ROM ",romInfo['Instance'].name, " is valid")
      else:
        allValid = False
    if allValid:
      self.raiseADebug("ROMs  are all valid for given model ", self.modelInstance.name)
    return allValid

  def submit(self,myInput,samplerType,jobHandler,**kwargs):
    """
        This will submit an individual sample to be evaluated by this model to a
        specified jobHandler. Note, some parameters are needed by createNewInput
        and thus descriptions are copied from there.
        @ In, myInput, list, the inputs (list) to start from to generate the new one
        @ In, samplerType, string, is the type of sampler that is calling to generate a new input
        @ In,  jobHandler, JobHandler instance, the global job handler instance
        @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
        @ Out, None
    """
    for mm in utils.returnImportModuleString(jobHandler):
      if mm not in self.mods:
        self.mods.append(mm)

    prefix = kwargs['prefix']
    self.tempOutputs['uncollectedJobIds'].append(prefix)
    self.counter = kwargs['counter']

    if not self.romConverged:
      trainingSize = self.checkTrainingSize()
      if trainingSize == self.romTrainStartSize:
        self.trainRom()
        self.romConverged = self.checkRomConvergence()
      elif trainingSize > self.romTrainStartSize and (trainingSize-self.romTrainStartSize)%self.romTrainStepSize == 0 and trainingSize <= self.romTrainMaxSize:
        self.trainRom()
        self.romConverged = self.checkRomConvergence()
      elif trainingSize > self.romTrainMaxSize:
        self.raiseAnError(IOError, "Maximum training size is reached, but ROMs are still not converged!")
    else:
      self.romValid = self.checkRomValidity()

    ## Ensemble models need access to the job handler, so let's stuff it in our
    ## catch all kwargs where evaluateSample can pick it up, not great, but
    ## will suffice until we can better redesign this whole process.
    kwargs['jobHandler'] = jobHandler

    ## This may look a little weird, but due to how the parallel python library
    ## works, we are unable to pass a member function as a job because the
    ## pp library loses track of what self is, so instead we call it from the
    ## class and pass self in as the first parameter
    jobHandler.addJob((self, myInput, samplerType, kwargs), self.__class__.evaluateSample, prefix, kwargs)

  def submitAsClient(self,myInput,samplerType,jobHandler,**kwargs):
    """
        This will submit an individual sample to be evaluated by this model to a
        specified jobHandler as a client job. Note, some parameters are needed
        by createNewInput and thus descriptions are copied from there.
        @ In, myInput, list, the inputs (list) to start from to generate the new
          one
        @ In, samplerType, string, is the type of sampler that is calling to
          generate a new input
        @ In,  jobHandler, JobHandler instance, the global job handler instance
        @ In, **kwargs, dict,  is a dictionary that contains the information
          coming from the sampler, a mandatory key is the sampledVars' that
          contains a dictionary {'name variable':value}
        @ Out, None
    """
    for mm in utils.returnImportModuleString(jobHandler):
      if mm not in self.mods:
        self.mods.append(mm)

    prefix = kwargs['prefix']
    self.tempOutputs['uncollectedJobIds'].append(prefix)
    self.counter = kwargs['counter']

    if not self.romConverged:
      trainingSize = self.checkTrainingSize()
      if trainingSize == self.romTrainStartSize:
        self.trainRom()
        self.romConverged = self.checkRomConvergence()
      elif trainingSize > self.romTrainStartSize and (trainingSize-self.romTrainStartSize)%self.romTrainStepSize == 0 and trainingSize <= self.romTrainMaxSize:
        self.trainRom()
        self.romConverged = self.checkRomConvergence()
      elif trainingSize > self.romTrainMaxSize:
        self.raiseAnError(IOError, "Maximum training size is reached, but ROMs are still not converged!")
    else:
      self.romValid = self.checkRomValidity()

    ## Ensemble models need access to the job handler, so let's stuff it in our
    ## catch all kwargs where evaluateSample can pick it up, not great, but
    ## will suffice until we can better redesign this whole process.
    kwargs['jobHandler'] = jobHandler

    ## This may look a little weird, but due to how the parallel python library
    ## works, we are unable to pass a member function as a job because the
    ## pp library loses track of what self is, so instead we call it from the
    ## class and pass self in as the first parameter
    jobHandler.addClientJob((self, myInput, samplerType, kwargs), self.__class__.evaluateSample, prefix, kwargs)

  def evaluateSample(self, myInput, samplerType, kwargs):
    """
        This will evaluate an individual sample on this model. Note, parameters
        are needed by createNewInput and thus descriptions are copied from there.
        @ In, myInput, list, the inputs (list) to start from to generate the new one
        @ In, samplerType, string, is the type of sampler that is calling to generate a new input
        @ In, kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
        @ Out, returnValue, dict, This holds the output information of the evaluated sample.
    """
    self.raiseADebug("Evaluate Sample")
    jobHandler = kwargs.pop('jobHandler')
    Input = self.createNewInput(myInput[0], samplerType, **kwargs)

    ## Unpack the specifics for this class, namely just the jobHandler
    returnValue = (Input,self._externalRun(Input,jobHandler))
    return returnValue

  def _externalRun(self,inRun, jobHandler):
    """
      Method that performs the actual run of the essembled model (separated from run method for parallelization purposes)
      @ In, inRun, tuple, tuple of Inputs (inRun[0] actual input, inRun[1] type of sampler,
        inRun[2] dictionary that contains information coming from sampler)
      @ In, jobHandler, instance, instance of jobHandler
      @ Out, exportDict, dict, dict of results from this hybrid model
    """
    self.raiseADebug("External Run")
    originalInput = inRun[0]
    samplerType = inRun[1]
    inputKwargs = inRun[2]
    identifier = inputKwargs.pop('prefix')
    uniqueHandler = self.name + identifier

    if self.romValid:
      # run roms
      exportDict = {}
      for romName, romInfo in self.romsDictionary.items():
        nextRom = False
        while not nextRom:
          if jobHandler.availability() > 0:
            romInfo['Instance'].submit(originalInput, samplerType, jobHandler, **inputKwargs[romName])
            while not jobHandler.isThisJobFinished(romName+utils.returnIdSeparator()+identifier):
              time.sleep(1.0e-3)
            nextRom = True
          else:
            time.sleep(1.0e-3)
          finishedRun = jobHandler.getFinished(jobIdentifier = romName+utils.returnIdSeparator()+identifier, uniqueHandler = uniqueHandler)
          evaluation = finishedRun[0].getEvaluation()
          if isinstance(evaluation, Runners.Error):
            self.raiseAnError(RuntimeError, "The rom "+romName+" identified by "+finishedRun[0].identifier+" failed!")
          # collect output in temporary data object
          tempExportDict = romInfo['Instance'].createExportDictionaryFromFinishedJob(finishedRun[0], True)
          self.__mergeDict(exportDict, tempExportDict)
    else:
      # run model
      inputKwargs['prefix'] = self.modelInstance.name+utils.returnIdSeparator()+identifier
      inputKwargs['uniqueHandler'] = self.name + identifier
      moveOn = False
      while not moveOn:
        if jobHandler.availability() > 0:
          self.modelInstance.submit(originalInput, samplerType, jobHandler, **inputKwargs)
          self.raiseADebug("Job submitted for model ", self.modelInstance.name, " with identifier ", identifier)
          while not jobHandler.isThisJobFinished(self.modelInstance.name+utils.returnIdSeparator()+identifier):
            time.sleep(1.0e-3)
          self.raiseADebug("Job finished ", self.modelInstance.name, " with identifier ", identifier)
          moveOn = True
        else:
          time.sleep(1.0e-3)
        finishedRun = jobHandler.getFinished(jobIdentifier = identifier, uniqueHandler = uniqueHandler)
        evaluation = finishedRun[0].getEvaluation()
        if isinstance(evaluation, Runners.Error):
          self.raiseAnError(RuntimeError, "The model "+self.modelInstance.name+" identified by "+finishedRun[0].identifier+" failed!")
        # collect output in temporary data object
        exportDict = self.modelInstance.createExportDictionaryFromFinishedJob(finishedRun[0], True)
        self.raiseADebug("Create exportDict")

    return exportDict

  def collectOutput(self,finishedJob,output):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, ClientRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    if isinstance(evaluation, Runners.Error):
      self.raiseAnError(RuntimeError,"Job " + finishedJob.identifier +" failed!")
    exportDict = evaluation[1]
    try:
      jobIndex = self.tempOutputs['uncollectedJobIds'].index(finishedJob.identifier)
      self.tempOutputs['uncollectedJobIds'].pop(jobIndex)
    except ValueError:
      jobIndex = None
    # save the data into targetEvaluation
    self.collectOutputFromDict(exportDict, self.targetEvaluationInstance)

    Dummy.collectOutput(self, finishedJob, output, options = {'exportDict':exportDict})

  def isRomConverged(self):
    """
    """
    converged = False

    converged = True
    return converged

  def isRomValid(self):
    """
    """
    valid = False

    valid = True
    return valid

  def checkTrainingSize(self):
    """
      This will check the size of existing training set
      @ In, None
      @ Out, existTrainingSize, int, the size of existing training set
    """
    existTrainingSize = self.counter - len(self.tempOutputs['uncollectedJobIds'])
    self.raiseADebug("Existing training size is: ", existTrainingSize)
    return existTrainingSize

  def __mergeDict(self,exportDict, tempExportDict):
    """
      This function will combine two dicts into one
      @ In, exportDict, dict, dictionary stores the input, output and metadata
      @ In, tempExportDict, dict, dictionary stores the input, output and metadata
    """
    if not exportDict:
      exportDict = copy.deepcopy(tempExportDict)
    else:
      inKey = 'inputSpaceParams'
      outKey = 'outputSpaceParams'
      for key, value in tempExportDict[inKey].items():
        exportDict[inKey][key] = value
      for key, value in tempExportDict[outKey].items():
        exportDict[outKey][key] = value
      for key, value in tempExportDict['metadata'].items():
        exportDict['metadata'][key] = value

    self.raiseADebug("The exportDict has been updated")


