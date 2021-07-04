#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:19:24 2021

@author: Yalin Li
"""

import os
data_path = os.path.join(os.path.dirname(__file__), 'data')
results_path = os.path.join(os.path.dirname(__file__), 'results')

from ._location import *
from ._ahp import *
from ._mcda import *
from ._bwaise import *

from . import (
    _location,
    _ahp,
    _mcda,
    _bwaise,
    )

__all__ = (
    'data_path',
    'results_path',
    *_location.__all__,
    *_ahp.__all__,
    *_mcda.__all__,
    )