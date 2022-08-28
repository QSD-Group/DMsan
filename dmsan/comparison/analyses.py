#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making of sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/DMsan/blob/main/LICENSE.txt
for license details.

Run this module to generate the data needed for figures.
'''

import os
from qsdsan.utils import save_pickle, load_pickle, time_printer
from dmsan.comparison import  results_path, figures_path
data = load_pickle(os.path.join(results_path, 'data.pckl'))