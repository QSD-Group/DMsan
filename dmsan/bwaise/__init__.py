#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:19:24 2021

@author: Yalin Li
"""

import os
bwaise_path = os.path.dirname(__file__)
scores_path = os.path.join(bwaise_path, 'scores')
results_path = os.path.join(bwaise_path, 'results')
figures_path = os.path.join(bwaise_path, 'figures')

# from .sys_simulation import *
# from .analysis import *
# from .line_graph import *

# from . import (
#     sys_simulation,
#     analysis,
#     line_graph,
#     )

# __all__ = (
#     'bwaise_path',
#     'scores_path',
#     'results_path',
#     'figures_path',
#     *sys_simulation.__all__,
#     )