#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:20:16 2021

@author: Yalin Li <zoe.yalin.li@gmail.com>

Run this script to look at the different possible scenarios that could change
the performance score of each system and thus the final winner.
"""

import os
import numpy as np, pandas as pd, seaborn as sns
from matplotlib import pyplot as plt
from qsdsan.utils import time_printer
from dmsan.bwaise import results_path, figures_path, import_from_pickle

loaded = import_from_pickle(param=False, tech_score=True, ahp=True, mcda=True,
                            uncertainty=False, sensitivity=None)

tech_score_dct = loaded['tech_score']
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
def run_across_indicator(mcda, alt, indicator, include_simulation_uncertainties=False,
                         min_val=None, max_val=None, step_num=10):
    '''Run all global weight scenarios at certain steps with the desired range.'''
    # Reset technology scores
    mcda.tech_scores = baseline_tech_scores
    idx = mcda.alt_names[mcda.alt_names==alt].index[0]

    min_val = min_val if min_val else 0
    max_val = max_val if max_val else baseline_tech_scores.loc[idx, indicator]

    # Total number of global weights
    weight_num = mcda.winners.shape[0]

    vals = np.linspace(min_val, max_val, num=step_num)
    win_dct = {}
    if not include_simulation_uncertainties: # fix other tech scores at the baseline

        for val in vals:
            mcda.tech_scores.loc[idx, indicator] = val
            mcda.run_MCDA()
            win_dct[val] = mcda.winners[mcda.winners.Winner==alt].shape[0]/weight_num

    else:
        # Make a copy for manipulating
        tech_score_dct_updated = tech_score_dct.copy()
        for val in vals:
            # Update all the costs to be the set value
            for v in tech_score_dct_updated.values():
                tech_score_dct_updated.loc[idx, indicator] = val
            score_df_dct, rank_df_dct, winner_df = \
                mcda.run_MCDA_multi_scores(tech_score_dct=tech_score_dct_updated)
            win_dct[val] = winner_df[winner_df==alt].count()/weight_num

    return win_dct


def plot_across_axis(win_dct, include_simulation_uncertainties=False):
    if not include_simulation_uncertainties:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(win_dct.keys(), win_dct.values())
    if include_simulation_uncertainties:
        dfs = []
        for val, win in win_dct.items():
            df_val = pd.DataFrame()
            df_val['win'] = win
            df_val['val'] = val
            dfs.append(df_val)
        df = pd.concat(dfs)
        df.reset_index(drop=True, inplace=True)
        ax = sns.lineplot(x=df['val'], y=df['win'])
    return ax




# %%

if __name__ == '__main__':
    # 100 steps take less than 1 min
    Cwin_across_cost = run_across_indicator(bwaise_mcda, alt='Alternative C',
                                            indicator='Econ1',
                                            include_simulation_uncertainties=False,
                                            min_val=0, step_num=100)
    ax0 = plot_across_axis(Cwin_across_cost, include_simulation_uncertainties=False)
    ax0.figure.savefig(os.path.join(figures_path, 'Cwin_across_cost.png'), dpi=100)
    
    # **Each** step takes ~1 hour (at each step, we are runnin 1000 simulation*1000 global weights)
    Cwin_across_cost_uncertainty = run_across_indicator(bwaise_mcda, alt='Alternative C',
                                                        indicator='Econ1',
                                                        include_simulation_uncertainties=True,
                                                        min_val=0, step_num=2)
    ax1 = plot_across_axis(Cwin_across_cost_uncertainty, include_simulation_uncertainties=True)
    ax1.figure.savefig(os.path.join(figures_path, 'Cwin_across_cost_with_band.png'), dpi=100)

    