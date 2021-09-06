#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yalin Li <zoe.yalin.li@gmail.com>
"""

import os
bwaise_path = os.path.dirname(__file__)
scores_path = os.path.join(bwaise_path, 'scores')
results_path = os.path.join(bwaise_path, 'results')
figures_path = os.path.join(bwaise_path, 'figures')
del os

from . import (
    sys_simulation,
    uncertainty_sensitivity,
    # scenario #!!! add this in once it's done
    )