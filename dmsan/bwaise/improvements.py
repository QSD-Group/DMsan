#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:20:16 2021

@author: Yalin Li <zoe.yalin.li@gmail.com>

Run this script to look at the different possible improvements that could change
the performance score of each system and thus the final winner.
"""

import os
import numpy as np, pandas as pd
from warnings import warn
from itertools import combinations, permutations
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from qsdsan.utils import time_printer, colors, save_pickle
from dmsan import AHP
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
baseline_local_weights = bwaise_ahp.norm_weights_df.copy()

def update_local_weights(tech_scores):
    ahp = AHP(location_name='Uganda', num_alt=bwaise_ahp.num_alt,
              na_default=0.00001, random_index={})

    # Set the local weight of indicators that all three systems score the same
    # to zero (to prevent diluting the scores)
    eq_ind = tech_scores.min()==tech_scores.max()
    eq_inds = [(i[:-1], i[-1]) for i in eq_ind[eq_ind==True].index]

    for i in eq_inds:
        # Need subtract in `int(i[1])-1` because of 0-indexing
        ahp.init_weights[i[0]][int(i[1])-1] = ahp.na_default

    ahp.get_AHP_weights(True)

    return ahp.norm_weights_df


# %%

# =============================================================================
# One-at-a-time test
# =============================================================================

# Best scores that the alternative can achieve,
# if None (meaning no absolute scale), will be chosen between the best
# (min for non-beneficial indicators and max for beneficial ones)
# among the three systems
best_score_dct = {
    'T1': 5,
    'T2': 3,
    'T3': 5,
    'T4': 7,
    'T6': 5,
    'T7': 2,
    'T8': 3,
    'T9': 3,
    'RR1': 1,
    'RR2': 1,
    'RR3': 1,
    'RR4': 1,
    'RR5': 1,
    'RR6': 1,
    'Env1': None,
    'Env2': None,
    'Env3': None,
    'Econ1': 0,
    'S1': 20,
    'S2': 12,
    'S3': 0, #!!! AHP weight for this indicator was set to 0 in `_ahp.py`, did we really mean that?
    'S4': 5,
    'S5': 1,
    'S6': 5,
    'S7': 0.1,
    }


@time_printer
def test_oat(mcda, alt, best_score={}):
    '''
    One-at-a-time test on if changing the tech score of one indicator would
    affect the overall winner.
    No uncertainties from system simulation are considered.

    If `all_at_once` is True, will change all the indicators at one time
    '''
    weight_num = mcda.criteria_weights.shape[0]
    alt_idx = mcda.alt_names[mcda.alt_names==alt].index[0]

    oat_dct = {ind: {} for ind in best_score.keys()}
    for ind, data in oat_dct.items():
        # Reset technology scores and refresh results
        mcda.tech_scores = baseline_tech_scores.copy()
        series = mcda.tech_scores.loc[:, ind]
        baseline = series.loc[alt_idx]
        ind_type = int(mcda.indicator_type[ind])
        baseline_rank = series.rank(
            ascending=bool(not ind_type), method='min').loc[alt_idx]

        if ind_type == 0: # non-beneficial
            if best_score.get(ind) is not None:
                updated = best_score[ind]
            else:
                updated = series.min()

        else: # beneficial
            if best_score.get(ind) is not None:
                updated = best_score[ind]
            else:
                updated = series.max()

        mcda.tech_scores.loc[alt_idx, ind] = updated
        updated_rank = mcda.tech_scores.loc[:, ind].rank(
            ascending=bool(not ind_type), method='min').loc[alt_idx]
        if updated_rank != 1:
            warn(f'The rank of indicator {ind} is {updated_rank} '
                 'with the provided best score.\n'
                 f'Scores for {ind} are:')
            print(mcda.tech_scores.loc[:, ind])

        # Update local weights
        mcda.indicator_weights = update_local_weights(mcda.tech_scores)

        # Run MCDA with multiple global weights
        mcda.run_MCDA()
        winning_chance = mcda.winners[mcda.winners.Winner==alt].shape[0]/weight_num

        data['indicator type']: ind_type
        data['baseline'] = baseline
        data['rank at baseline'] = baseline_rank
        data['updated'] = updated
        data['rank after updating'] = updated_rank
        data['winning chance'] = winning_chance

    # Get the winning chance at baseline values
    mcda.tech_scores = baseline_tech_scores.copy()
    mcda.indicator_weights = baseline_local_weights.copy()
    mcda.run_MCDA()
    oat_dct['baseline'] = {'winning chance':
                            mcda.winners[mcda.winners.Winner==alt].shape[0]/weight_num}

    return oat_dct


def plot_oat(oat_dct, file_path=''):
    labels, winning = ['baseline'], [oat_dct['baseline']['winning chance']]
    get_rounded = lambda val: round(val, 2) if len(str(val).split('.')[-1]) >= 2 else val

    for ind, data in oat_dct.items():
        if not ind == 'baseline':
        # if not ind in ('baseline', 'all at once'):
            labels.append(f'{ind}: {get_rounded(data["baseline"])}->{get_rounded(data["updated"])}')
            winning.append(data['winning chance'])

    fig, ax = plt.subplots(figsize=(6, 8))
    y = np.arange(len(labels))
    ax.barh(y, winning, align='center')
    ax.set(yticks=y, yticklabels=labels, ylabel='Changed indicator (one-at-a-time)',
           xlabel='Winning chance')
    ax.invert_yaxis()  # labels read top-to-bottom

    fig.subplots_adjust(left=0.5)

    if file_path is not None:
        file_path = file_path if file_path != '' \
            else os.path.join(figures_path, 'baseline_oat.png')
        fig.savefig(file_path, dpi=100)

    return ax


# %%

# =============================================================================
# Find the local optimum
# =============================================================================

@time_printer
def local_optimum_approach(mcda, alt, oat_dct, file_path=''):
    '''Find the local optimum trajectory for improving the indicator scores.'''
    weight_num = mcda.criteria_weights.shape[0]
    alt_idx = mcda.alt_names[mcda.alt_names==alt].index[0]

    winning_chances = [data['winning chance'] for data in oat_dct.values()]
    loc_dct = {ind: data['winning chance']
               for ind, data in oat_dct.items()
               if data['winning chance']==max(winning_chances)}
    ind = list(loc_dct.keys())[0]
    winning_chance = list(loc_dct.values())[0]
    updated_scores = baseline_tech_scores.copy()
    updated_scores.loc[alt_idx, ind] = winning_chance

    copied = oat_dct.copy()
    copied.pop(ind)['winning chance']
    loc_dct['baseline'] = copied.pop('baseline')['winning chance']

    # End looping if already has reached 100% winning
    # or tried all indicators
    n = 2
    loc_df = pd.DataFrame(index=[i for i in oat_dct.keys() if i!='baseline'])
    loc_df[1] = winning_chances[:-1]
    while (winning_chance<1 and len(copied)!=0):
        winning_dct = {}
        for ind, data in copied.items():
            mcda.tech_scores = updated_scores.copy()
            mcda.tech_scores.loc[alt_idx, ind] = data['updated']

            # Update local weights
            mcda.indicator_weights = update_local_weights(mcda.tech_scores)

            # Run MCDA with multiple global weights
            mcda.run_MCDA()
            winning_dct[ind] = mcda.winners[mcda.winners.Winner==alt].shape[0]/weight_num

        series = pd.Series(data=winning_dct.values(), index=winning_dct.keys())
        loc_df[n] = series
        n += 1

        # Find the indicator change with the highest winning chance
        best_chance = max(winning_dct.values())
        best_ind = list(winning_dct.keys())[list(winning_dct.values()).index(best_chance)]
        updated_scores.loc[alt_idx, best_ind] = oat_dct[best_ind]['updated']
        winning_chance = loc_dct[best_ind] = best_chance
        copied.pop(best_ind)

    if file_path is not None:
        file_path = file_path if file_path != '' \
            else os.path.join(results_path, 'improvements/local_optimum.xlsx')
        loc_df.to_excel(file_path)

    return loc_dct, loc_df


def plot_acc(acc_dct, file_path=None):
    fig, ax = plt.subplots(figsize=(6, 8))
    labels = [f'+{i}' for i in acc_dct.keys()]
    values = list(acc_dct.values())
    if '+baseline' in labels:
        idx = labels.index('+baseline')
        labels.insert(0, labels.pop(idx))
        values.insert(0, values.pop(idx))

    labels[0] = labels[0].lstrip('+')

    x = np.arange(len(labels))
    ax.plot(x, values, '-o')

    ax.set(xticks=x, xticklabels=labels, xlabel='Changed indicator (accumulated)',
            ylabel='Winning chance')

    for label in ax.get_xticklabels():
        label.set_rotation(30)

    if file_path is not None:
        file_path = file_path if file_path != '' \
            else os.path.join(figures_path, 'baseline_acc.png')
        fig.savefig(file_path, dpi=100)

    return ax


def plot_local_optimum(loc_dct, file_path=''):
    fig, ax = plt.subplots(figsize=(6, 8))
    labels = [f'+{i}' for i in loc_dct.keys()]
    values = list(loc_dct.values())
    if '+baseline' in labels:
        idx = labels.index('+baseline')
        labels.insert(0, labels.pop(idx))
        values.insert(0, values.pop(idx))

    labels[0] = labels[0].lstrip('+')

    x = np.arange(len(labels))
    ax.plot(x, values, '-o')

    ax.set(xticks=x, xticklabels=labels, xlabel='Changed indicator (accumulated)',
            ylabel='Winning chance')

    for label in ax.get_xticklabels():
        label.set_rotation(30)

    if file_path is not None:
        file_path = file_path if file_path != '' \
            else os.path.join(figures_path, 'local_optimum.png')
        ax.figure.savefig(file_path, dpi=100)
    return ax


# %%

# =============================================================================
# Test all possible improvements to find the global optimum
# =============================================================================

@time_printer
def global_optimum_approach(mcda, alt, oat_dct,
                            consider_order=False, select_top=None,
                            target_chance=1, cutoff_step=None, file_path=''):
    '''
    Test all possible routes to the targeted winning chance.

    If `select_top` is provided (int), will only look at the X-best indicators
    at baseline.

    If a `cutoff_step` is provided, routes that haven't reached 100% of winning chance
    when `cutoff_step` number of technology improvements have been made
    would be stopped at the `cutoff_step`.

    Otherwise, the `cutoff_step` will be the same as the number of indicators
    where the alternative has not achieved the best score.
    '''
    weight_num = mcda.criteria_weights.shape[0]
    alt_idx = mcda.alt_names[mcda.alt_names==alt].index[0]
    # updated_scores = baseline_tech_scores.copy()

    # Take care of the baseline winning chance
    copied = oat_dct.copy()
    baseline_chance = copied.pop('baseline')['winning chance']

    # Eliminate the indicators whose updated values are the same as baseline values
    already_best = [i for i in copied.keys() \
                    if copied[i]['baseline']==copied[i]['updated']]
    if already_best:
        print(f'Indicators {", ".join(already_best)} have already achieved the best score, '
              'will be excluded from the global optimum test.')
        for ind in already_best: # exclude from the test
            copied.pop(ind)

    # Only look at the top `select_top` indicators, if `select_top` is provided
    copied_df = pd.DataFrame.from_dict(copied).transpose()
    copied_df.sort_values('winning chance', ascending=False, inplace=True)
    inds = copied_df.index.to_list()[:select_top] if select_top else copied_df.index.to_list()

    # Permutations to be iterated from
    cutoff_step = cutoff_step or len(inds) # exhaust all indicators if `cutpf_step` not provided
    runs = list(combinations(inds, cutoff_step)) if not consider_order \
        else list(permutations(inds, cutoff_step))

    glob_dct = {'baseline': {1:baseline_chance}}
    for run in runs:
        glob_dct[run] = {}
        mcda.tech_scores = baseline_tech_scores.copy()
        winning_chance = 0
        n = 1
        for ind in run:
            mcda.tech_scores.loc[alt_idx, ind] = copied_df.loc[ind]['updated']
            if n == 1: # at baseline, no need to run again
                glob_dct[run][n] = winning_chance = \
                    copied_df.loc[ind]['winning chance']
            else:
                # Update local weights
                mcda.indicator_weights = update_local_weights(mcda.tech_scores)
                # Run MCDA with multiple global weights
                mcda.run_MCDA()
                glob_dct[run][n] = winning_chance = \
                    mcda.winners[mcda.winners.Winner==alt].shape[0]/weight_num
            if winning_chance >= target_chance:
                break
            n += 1

    glob_df = pd.DataFrame.from_dict(glob_dct).transpose()

    save_pickle(glob_dct, os.path.join(results_path, 'improvements/glob_dct.pckl'))

    if file_path is not None:
        file_path = file_path if file_path != '' \
            else os.path.join(results_path, 'improvements/global_optimum.xlsx')
        glob_df.to_excel(file_path)

    return glob_dct, glob_df


# Colors for plotting
Guest_colors = colors.Guest
color_dct = {
    'T': Guest_colors.red.RGBn,
    'RR': Guest_colors.yellow.RGBn,
    'Econ': Guest_colors.green.RGBn,
    'Env': Guest_colors.blue.RGBn,
    'S': Guest_colors.purple.RGBn,
    }
get_colors = lambda inds: [color_dct[ind[:-1]] for ind in inds]

def plot_global_optimum(glob_dct, file_path=''):
    fig, ax = plt.subplots(figsize=(6, 8))
    for idx, chance_dct in glob_dct.items():
        x = list(chance_dct.keys())
        y = list(chance_dct.values())

        # # All the same color
        # ax.plot(x, y, '-k', linewidth=0.5)

        # Use different colors
        if idx == 'baseline':
            ax.plot(x, y, '-k', linewidth=1.5)
        else: # plot the line graph in segments with different colors
            c = get_colors(idx)
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=c)
            lc.set_array(x[:1])
            lc.set_linewidth(0.5)
            ax.add_collection(lc)

    ax.set(xticks=x, xlabel='Number of changed indicator',
           yticks=[0, 0.25, 0.5, 0.75, 1], ylabel='Winning chance')

    if file_path is not None:
        file_path = file_path if file_path != '' \
            else os.path.join(figures_path, 'global_optimum.png')
        ax.figure.savefig(file_path, dpi=300)
    return ax


# %%

if __name__ == '__main__':
    oat_dct = test_oat(bwaise_mcda, alt='Alternative C', best_score=best_score_dct)
    ax = plot_oat(oat_dct, file_path='')

    loc_dct, loc_df = local_optimum_approach(
        bwaise_mcda, alt='Alternative C', oat_dct=oat_dct, file_path='')
    ax = plot_local_optimum(loc_dct, file_path='')
    glob_dct, glob_df = global_optimum_approach(
        bwaise_mcda, 'Alternative C', oat_dct, consider_order=False,
        select_top=10, target_chance=1, cutoff_step=len(loc_dct)-1) # subtract 1 for baseline
    ax = plot_global_optimum(glob_dct)


# %%

# =============================================================================
# If want to consider uncertainties with different criteria weight scenarios
# =============================================================================

# import seaborn as sns

# @time_printer
# def test_across_axis(mcda, alt, indicator, include_simulation_uncertainties=False,
#                      min_val=None, max_val=None, step_num=10):
#     '''Run all global weight scenarios at certain steps with the desired range.'''
#     # Reset technology scores
#     mcda.tech_scores = baseline_tech_scores.copy()
#     idx = mcda.alt_names[mcda.alt_names==alt].index[0]

#     min_val = min_val if min_val else 0
#     max_val = max_val if max_val else baseline_tech_scores.loc[idx, indicator]

#     # Total number of global weights
#     weight_num = mcda.criteria_weights.shape[0]

#     vals = np.linspace(min_val, max_val, num=step_num)
#     win_dct = {}
#     if not include_simulation_uncertainties: # fix other tech scores at the baseline

#         for val in vals:
#             mcda.tech_scores.loc[idx, indicator] = val
#             mcda.run_MCDA()
#             win_dct[val] = mcda.winners[mcda.winners.Winner==alt].shape[0]/weight_num

#     else:
#         # Make a copy for manipulating
#         tech_score_dct_updated = tech_score_dct.copy()
#         for val in vals:
#             # Update all the costs to be the set value
#             for v in tech_score_dct_updated.values():
#                 tech_score_dct_updated.loc[idx, indicator] = val
#             score_df_dct, rank_df_dct, winner_df = \
#                 mcda.run_MCDA_multi_scores(tech_score_dct=tech_score_dct_updated)
#             win_dct[val] = winner_df[winner_df==alt].count()/weight_num

#     return win_dct


# def plot_across_axis(win_dct, include_simulation_uncertainties=False):
#     if not include_simulation_uncertainties:
#         fig, ax = plt.subplots(figsize=(8,4.5))
#         ax.plot(win_dct.keys(), win_dct.values())
#     if include_simulation_uncertainties:
#         dfs = []
#         for val, win in win_dct.items():
#             df_val = pd.DataFrame()
#             df_val['win'] = win
#             df_val['val'] = val
#             dfs.append(df_val)
#         df = pd.concat(dfs)
#         df.reset_index(drop=True, inplace=True)
#         ax = sns.lineplot(x=df['val'], y=df['win'])
#     return ax

# if __name__ == '__main__':
#     # 100 steps take less than 1 min
#     Cwin_across_cost = test_across_axis(bwaise_mcda, alt='Alternative C',
#                                         indicator='Econ1',
#                                         include_simulation_uncertainties=False,
#                                         min_val=0, step_num=100)
#     ax0 = plot_across_axis(Cwin_across_cost, include_simulation_uncertainties=False)
#     ax0.figure.savefig(os.path.join(figures_path, 'Cwin_across_cost.png'), dpi=100)

#     # **Each** step takes ~1 hour (at each step, we are running 1000 simulation*1000 global weights)
#     Cwin_across_cost_uncertainty = test_across_axis(bwaise_mcda, alt='Alternative C',
#                                                     indicator='Econ1',
#                                                     include_simulation_uncertainties=True,
#                                                     min_val=0, step_num=2)
#     ax1 = plot_across_axis(Cwin_across_cost_uncertainty, include_simulation_uncertainties=True)
#     ax1.figure.savefig(os.path.join(figures_path, 'Cwin_across_cost_with_band.png'), dpi=100)


# %%

# # Legacy codes to test accumulatice effects of improving indicators
# # might be removed in the future because the results do not mean much

# def test_acc(mcda, alt, oat_dct):
#     '''Test the accumulative effects of the one-at-a-time best (at baseline).'''
#     acc_dct = {}
#     baseline_winning = oat_dct['baseline']['winning chance']
#     for ind, data_dct in oat_dct.items():
#         ind_winning = data_dct['winning chance']
#         if (ind_winning == baseline_winning or ind_winning == 1):
#             continue

#         acc_dct[ind] = ind_winning

#     # Sort the dict in order
#     ind_list = [i for i in acc_dct.keys() if not i in ('baseline', 'all at once')]
#     ind_list.sort(key=lambda ind: acc_dct[ind], reverse=True)
#     ind_list.insert(0, 'baseline')
#     acc_dct['baseline'] = baseline_winning
#     acc_dct = {ind:acc_dct[ind] for ind in ind_list}

#     weight_num = mcda.criteria_weights.shape[0]
#     idx = mcda.alt_names[mcda.alt_names==alt].index[0]

#     for ind in acc_dct.keys():
#         if ind == 'baseline':
#             pass
#         else:
#             mcda.tech_scores.loc[idx, ind] = oat_dct[ind]['updated']
#             mcda.indicator_weights = update_local_weights(mcda.tech_scores)
#             mcda.run_MCDA()
#             acc_dct[ind] = mcda.winners[mcda.winners.Winner==alt].shape[0]/weight_num

#     return acc_dct