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

def test_oat(mcda, alt, indicators=None, method='best'):
    '''
    One-at-a-time test on if changing the tech score of one indicator would
    affect the overall winner.
    No uncertainties from system simulation are considered.

    If `method`=='best', then update the indicator score to be the best score;
    if `method` is a number, then the score will be changed to (1-method)
    for non-beneficial indicators or (1+method) for beneficial indicators.
    '''

    weight_num = mcda.criteria_weights.shape[0]
    idx = mcda.alt_names[mcda.alt_names==alt].index[0]

    if indicators is None:
        indicators = mcda.indicator_type.columns
    else:
        try:
            iter(indicators)
        except TypeError:
            indicators = (indicators,)

    test_dct = {ind: {} for ind in indicators}
    for ind, data in test_dct.items():
        # Reset technology scores and refresh results
        mcda.tech_scores = baseline_tech_scores.copy()
        series = mcda.tech_scores.loc[:, ind]
        baseline = series.loc[idx]
        ind_type = int(mcda.indicator_type[ind])
        baseline_rank = series.rank(ascending=bool(not ind_type)).loc[idx]
        if baseline_rank < 2:
            print(f'{alt} already winning indicator {ind}.')

        if ind_type == 0: # non-beneficial
            if method == 'best':
                updated = series.min()
            else:
                updated = baseline * (1-method)

        else: # beneficial
            if method == 'best':
                updated = series.max()
            else:
                updated = baseline * (1+method)

        mcda.tech_scores.loc[idx, ind] = updated
        updated_rank = series.rank(ascending=bool(not ind_type)).loc[idx]

        # Run MCDA with multiple global weights
        mcda.run_MCDA()
        win_chance = mcda.winners[mcda.winners.Winner==alt].shape[0]/weight_num

        data['indicator type']: ind_type
        data['baseline'] = baseline
        data['rank at baseline'] = baseline_rank
        data['updated'] = updated
        data['rank after updating'] = updated_rank
        data['winning chance'] = win_chance

    # Get the winning chance at baseline values
    mcda.tech_scores = baseline_tech_scores.copy()
    mcda.run_MCDA()
    test_dct['baseline'] = {'winning chance':
                            mcda.winners[mcda.winners.Winner==alt].shape[0]/weight_num}

    return test_dct


def plot_oat(test_dct):
    labels, winning = ['baseline'], [test_dct['baseline']['winning chance']]
    get_rounded = lambda val: round(val, 2) if len(str(val).split('.')[-1]) >= 2 else val

    for ind, data in test_dct.items():
        if not ind == 'baseline':
            labels.append(f'{ind}: {get_rounded(data["baseline"])}->{get_rounded(data["updated"])}')
            winning.append(data['winning chance'])

    fig, ax = plt.subplots(figsize=(6, 8))
    y = np.arange(len(labels))
    ax.barh(y, winning, align='center')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Winning chance')

    fig.subplots_adjust(left=0.5)

    return ax


# %%

@time_printer
def test_across_axis(mcda, alt, indicator, include_simulation_uncertainties=False,
                     min_val=None, max_val=None, step_num=10):
    '''Run all global weight scenarios at certain steps with the desired range.'''
    # Reset technology scores
    mcda.tech_scores = baseline_tech_scores.copy()
    idx = mcda.alt_names[mcda.alt_names==alt].index[0]

    min_val = min_val if min_val else 0
    max_val = max_val if max_val else baseline_tech_scores.loc[idx, indicator]

    # Total number of global weights
    weight_num = mcda.criteria_weights.shape[0]

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
        fig, ax = plt.subplots(figsize=(8,4.5))
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
    # # 100 steps take less than 1 min
    # Cwin_across_cost = test_across_axis(bwaise_mcda, alt='Alternative C',
    #                                     indicator='Econ1',
    #                                     include_simulation_uncertainties=False,
    #                                     min_val=0, step_num=100)
    # ax0 = plot_across_axis(Cwin_across_cost, include_simulation_uncertainties=False)
    # ax0.figure.savefig(os.path.join(figures_path, 'Cwin_across_cost.png'), dpi=100)

    # # **Each** step takes ~1 hour (at each step, we are runnin 1000 simulation*1000 global weights)
    # Cwin_across_cost_uncertainty = test_across_axis(bwaise_mcda, alt='Alternative C',
    #                                                 indicator='Econ1',
    #                                                 include_simulation_uncertainties=True,
    #                                                 min_val=0, step_num=2)
    # ax1 = plot_across_axis(Cwin_across_cost_uncertainty, include_simulation_uncertainties=True)
    # ax1.figure.savefig(os.path.join(figures_path, 'Cwin_across_cost_with_band.png'), dpi=100)

    test_dct_best = test_oat(bwaise_mcda, alt='Alternative C', method='best')
    ax = plot_oat(test_dct_best)
    ax.figure.savefig(os.path.join(figures_path, 'test_best.png'), dpi=100)

    test_dct_100 = test_oat(bwaise_mcda, alt='Alternative C', method=1)
    ax = plot_oat(test_dct_100)
    ax.figure.savefig(os.path.join(figures_path, 'test_100.png'), dpi=100)