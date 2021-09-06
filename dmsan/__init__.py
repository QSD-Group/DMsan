#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yalin Li <zoe.yalin.li@gmail.com>
"""

import os
path = os.path.dirname(__file__)
data_path = os.path.join(path, 'data')
results_path = os.path.join(path, 'results')

from ._location import *
from ._ahp import *
from ._mcda import *

from . import (
    _location,
    _ahp,
    _mcda,
    )

__all__ = (
    'path',
    'data_path',
    'results_path',
    *_location.__all__,
    *_ahp.__all__,
    *_mcda.__all__,
    )