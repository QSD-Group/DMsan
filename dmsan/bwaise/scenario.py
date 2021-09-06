#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:20:16 2021

@author: Yalin Li <zoe.yalin.li@gmail.com>

Run this script to look at the different possible scenarios that could change
the performance score of each system and thus the final winner.
"""

from dmsan.bwaise import results_path
from dmsan.bwaise.uncertainty_sensitivity import (
    import_from_pickle, generate_weights,
    )


loaded = import_from_pickle(ahp=True, mcda=True,
                            uncertainty=False, sensitivity=None)

bwaise_ahp = loaded['ahp']
bwaise_mcda = loaded['mcda']