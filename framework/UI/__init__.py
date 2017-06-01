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
  The UI module includes the different user interfaces available within RAVEN.

  Created on November 30, 2016
  @author: maljdp
"""

from __future__ import absolute_import

## These lines ensure that we do not have to do something like:
## 'from UI.Window import Window' outside
## of this submodule
from .ZoomableGraphicsView import ZoomableGraphicsView
from .BaseHierarchicalView import BaseHierarchicalView
from .DendrogramView import DendrogramView
from . import colors
from .FitnessView import FitnessView
from .ScatterView2D import ScatterView2D
from .ScatterView3D import ScatterView3D
from .SensitivityView import SensitivityView
from .TopologyMapView import TopologyMapView
from .HierarchyWindow import HierarchyWindow
from .TopologyWindow import TopologyWindow
from .DataInterpreterDialog import DataInterpreterDialog

## As these are not exposed to the user, we do not need a factory to dynamically
## allocate them. They will be explicitly called when needed everywhere in the
## code.
# from .Factory import knownTypes
# from .Factory import returnInstance
# from .Factory import returnClass

# We should not really need this as we do not use wildcard imports
__all__ = ['colors', 'HierarchylWindow', 'DendrogramView',
           'TopologyWindow', 'FitnessView', 'ScatterView2D',
           'ScatterView3D', 'SensitivityView', 'TopologyMapView']
