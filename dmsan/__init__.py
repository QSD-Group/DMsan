#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making for sanitation and resource recovery systems

This module is developed by:
    Yalin Li <mailto.yalin.li@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/DMsan/blob/main/LICENSE.txt
for license details.
'''
import os
path = os.path.dirname(__file__)
data_path = os.path.join(path, 'data')

# Order matters
from ._location import *
from ._ahp import *
from ._mcda import *
from .utils import *

# Order doesn't matter
from . import (
    _ahp,
    _location,
    _mcda,
    utils,
    )

__all__ = (
    *_ahp.__all__,
    *_location.__all__,
    *_mcda.__all__,
    'data_path',
    'path',
    *utils.__all__,
    )