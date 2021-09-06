#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:20:16 2021

@author: Yalin Li <zoe.yalin.li@gmail.com>

Run this script to look at the different possible scenarios that could change
the performance score of each system and thus the final winner.
"""

import os
import numpy as np, pandas as pd
from matplotlib import pyplot as plt
from qsdsan.utils import time_printer
from dmsan.bwaise import results_path, import_from_pickle

loaded = import_from_pickle(ahp=True, mcda=True,
                            uncertainty=False, sensitivity=None)
bwaise_ahp = loaded['ahp']
bwaise_mcda = loaded['mcda']

file_path = os.path.join(results_path, 'Global_weights.xlsx')
bwaise_mcda.criteria_weights = pd.read_excel(file_path)

# Save a copy
baseline_tech_scores =  bwaise_mcda.tech_scores.copy()


# %%

# =============================================================================
# Different scenairos
# =============================================================================

@time_printer
def run_across_indicator(mcda, alt, indicator,
                         min_val=None, max_val=None, step_num=10):
    '''Run all global weight scenarios at certain steps with the desired range.'''
    # Reset technology scores
    mcda.tech_scores = baseline_tech_scores
    idx = mcda.alt_names[mcda.alt_names==alt].index[0]

    min_val = min_val if min_val else 0
    max_val = max_val if max_val else baseline_tech_scores.loc[idx, indicator]

    # Total number of global weights
    weight_num = mcda.winners.shape[0]

    # Update technology score and rerun MCDA
    vals = np.linspace(min_val, max_val, num=step_num)
    won = {}
    for val in vals:
        mcda.tech_scores.loc[idx, indicator] = val
        mcda.run_MCDA()
        won[val] = mcda.winners[mcda.winners.Winner==alt].shape[0]/weight_num

    return won


def plot_across_axis(won_dct):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(won_dct.keys(), won_dct.values())
    return fig, ax


if __name__ == '__main__':
    Cwon_across_cost = run_across_indicator(bwaise_mcda, alt='Alternative C',
                                            indicator='Econ1',
                                            min_val=0, step_num=100)
    fig, ax = plot_across_axis(Cwon_across_cost)