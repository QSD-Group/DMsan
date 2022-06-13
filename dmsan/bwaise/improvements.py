#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:20:16 2021

@author: Yalin Li <mailto.yalin.li@gmail.com>

Run this script to look at the different possible improvements that could change
the performance score of each system and thus the final winner.

There are a total of 28 indicators for the five criteria,
26 of which are included in the analysis (S8 and S9 excluded).

Alternative C achieved the best score for four indicators (T3, T5, T7, Env1)
without any improvements.
"""

import os
import numpy as np, pandas as pd, seaborn as sns
from warnings import warn
from itertools import combinations, permutations
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from qsdsan.utils import time_printer, colors, save_pickle
from dmsan.bwaise import results_path, figures_path, import_from_pickle
from dmsan.bwaise.uncertainty_sensitivity import \
    criterion_num, wt_scenario_num as sce_num1


loaded = import_from_pickle(parameters=False, indicator_scores=True,
                            ahp=True, mcda=True,
                            uncertainty=False, sensitivity=None)

ind_score_dct = loaded['indicator_scores']
bwaise_ahp = loaded['ahp']
bwaise_mcda = loaded['mcda']

# Save a copy
baseline_indicator_scores =  bwaise_mcda.indicator_scores.copy()
baseline_indicator_weights = bwaise_ahp.norm_weights_df.copy()

# # DO NOT DELETE
# # Legacy code to set the local weight of indicators
# # that all three systems score the same to zero
# from dmsan import AHP
# def update_indicator_weights(ind_scores):
#     ahp = AHP(location_name='Uganda', num_alt=bwaise_ahp.num_alt,
#               na_default=0.00001, random_index={})
#     eq_ind = ind_scores.min()==ind_scores.max()
#     eq_inds = eq_ind[eq_ind==True].index
#     for i in eq_inds:
#         bwaise_ahp.init_weights[i] = bwaise_ahp.na_default
#     ahp.get_indicator_weights(return_results=True)
#     return ahp.norm_weights_df


# %%

# =============================================================================
# One-at-a-time test
# =============================================================================

# Best scores that the alternative can achieve,
# if None (meaning no absolute scale), will be chosen between the best
# (min for non-beneficial indicators and max for beneficial ones)
# among the three systems
# Ones that were commented out were not included in the paper,
# but the values nonetheless represent the best possible values
best_score_dct = {
    # 'T1': 5,
    # 'T2': 3,
    # 'T3': 5,
    # 'T4': 7,
    # 'T5': 5,
    # 'T6': 5,
    'T7': 3,
    'T8': 3,
    'T9': 3, # note that T9 and RR1 both represent water stress and should be improved together
    'RR1': 1,
    'RR2': 1,
    'RR3': 1,
    'RR4': 1,
    # 'RR5': 1,
    # 'RR6': 1,
    'Env1': None,
    'Env2': None,
    'Env3': None,
    'Econ1': None,
    'S1': 24, # total job = baseline of 12 + high paying job of 12
    'S2': 12,
    # 'S3': 0,
    # 'S4': 5,
    # 'S5': 1,
    'S6': 5,
    }


@time_printer
def test_oat(mcda, alt, best_score={}):
    '''
    One-at-a-time test on if changing the tech score of one indicator would
    affect the overall winner.
    No uncertainties from system simulation are considered.

    If `all_at_once` is True, will change all the indicators at one time
    '''
    weight_num = mcda.criterion_weights.shape[0]
    alt_idx = mcda.alt_names[mcda.alt_names==alt].index[0]

    oat_dct = {ind: {} for ind in best_score.keys()}
    for ind, data in oat_dct.items():
        if ind == 'RR1': continue # RR1 and T9 are the same
        # Reset technology scores and refresh results
        mcda.indicator_scores = baseline_indicator_scores.copy()
        series = mcda.indicator_scores.loc[:, ind]
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

        mcda.indicator_scores.loc[alt_idx, ind] = updated
        if ind == 'T9': # T9 and RR1 are the same
            mcda.indicator_scores.loc[alt_idx, 'RR1'] = best_score_dct['RR1']
        updated_rank = mcda.indicator_scores.loc[:, ind].rank(
            ascending=bool(not ind_type), method='min').loc[alt_idx]
        if updated_rank != 1:
            warn(f'The rank of indicator {ind} is {updated_rank} '
                 'with the provided best score.\n'
                 f'Scores for {ind} are:')
            print(mcda.indicator_scores.loc[:, ind])

        # # DO NOT DELETE
        # # Legacy code to update local weights
        # mcda.indicator_weights = update_indicator_weights(mcda.indicator_scores)

        # Run MCDA with multiple global weights
        mcda.run_MCDA(file_path=None)
        winning_chance = mcda.winners[mcda.winners.Winner==alt].shape[0]/weight_num

        data['indicator type']: ind_type
        data['baseline'] = baseline
        data['rank at baseline'] = baseline_rank
        data['updated'] = updated
        data['rank after updating'] = updated_rank
        data['winning chance'] = winning_chance

    # Get the winning chance at baseline values
    mcda.indicator_scores = baseline_indicator_scores.copy()
    mcda.indicator_weights = baseline_indicator_weights.copy()
    mcda.run_MCDA(file_path=None)
    oat_dct['baseline'] = {'winning chance':
                            mcda.winners[mcda.winners.Winner==alt].shape[0]/weight_num}

    return oat_dct


def plot_oat(oat_dct, wt_sce_num, file_path=''):
    labels, winning = ['baseline'], [oat_dct['baseline']['winning chance']]
    get_rounded = lambda val: round(val, 2) if len(str(val).split('.')[-1]) >= 2 else val

    for ind, data in oat_dct.items():
        if ind == 'RR1': continue # the same as T9
        if not ind == 'baseline':
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
            else os.path.join(figures_path, f'improvements_oat_{wt_sce_num}.png')
        fig.savefig(file_path, dpi=300)

    return ax


# %%

# =============================================================================
# Find and plot the local optimum
# =============================================================================

@time_printer
def local_optimum_approach(mcda, alt, oat_dct, wt_sce_num, file_path=''):
    '''Find the local optimum trajectory for improving the indicator scores.'''
    weight_num = mcda.criterion_weights.shape[0]
    alt_idx = mcda.alt_names[mcda.alt_names==alt].index[0]

    winning_chances = [data['winning chance'] for data in oat_dct.values()]
    loc_dct = {ind: data['winning chance']
               for ind, data in oat_dct.items()
               if data['winning chance']==max(winning_chances)}
    ind = list(loc_dct.keys())[0]
    winning_chance = list(loc_dct.values())[0]
    updated_scores = baseline_indicator_scores.copy()
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
            mcda.indicator_scores = updated_scores.copy()
            mcda.indicator_scores.loc[alt_idx, ind] = data['updated']

            # # DO NOT DELETE
            # # Legacy code to update local weights
            # mcda.indicator_weights = update_indicator_weights(mcda.indicator_scores)

            # Run MCDA with multiple global weights
            mcda.run_MCDA(file_path=None)
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
            else os.path.join(results_path, f'improvements/local_optimum_{wt_sce_num}.xlsx')
        loc_df.to_excel(file_path)

    return loc_dct, loc_df


def plot_local_optimum(loc_dct, wt_sce_num, file_path=''):
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
    ax.grid(axis='x', color=Guest_colors.gray.RGBn, linestyle='--', linewidth=0.5)

    for label in ax.get_xticklabels():
        label.set_rotation(30)

    if file_path is not None:
        file_path = file_path if file_path != '' \
            else os.path.join(figures_path, f'improvements_local_{wt_sce_num}.png')
        ax.figure.savefig(file_path, dpi=300)
    return ax


# %%

# =============================================================================
# Test all possible improvements to find the global optimum
# =============================================================================

@time_printer
def global_optimum_approach(mcda, alt, oat_dct, wt_sce_num,
                            select_top=None, target_chance=1,
                            cutoff_step=None, file_path=''):
    '''
    Test all possible routes to the target winning chance.

    If `select_top` is provided (int), will only look at the X-best indicators
    at baseline.

    If a `cutoff_step` is provided, routes that haven't reached 100% of winning chance
    when `cutoff_step` number of technology improvements have been made
    would be stopped at the `cutoff_step`.

    Otherwise, the `cutoff_step` will be the same as the number of indicators
    where the alternative has not achieved the best score.
    '''
    weight_num = mcda.criterion_weights.shape[0]
    alt_idx = mcda.alt_names[mcda.alt_names==alt].index[0]

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

    # Function to run through the given iterations
    def iter_run(runs):
        glob_dct = {'baseline': {0: baseline_chance}}
        for run in runs:
            glob_dct[run] = {}
            mcda.indicator_scores = baseline_indicator_scores.copy()
            winning_chance = 0
            n = 1
            for ind in run:
                mcda.indicator_scores.loc[alt_idx, ind] = copied_df.loc[ind]['updated']
                if n == 1: # at baseline, no need to run again
                    glob_dct[run][n] = winning_chance = \
                        copied_df.loc[ind]['winning chance']
                else:
                    # # DO NOT DELETE
                    # # Legacy code to update local weights
                    # mcda.indicator_weights = update_indicator_weights(mcda.indicator_scores)

                    # Run MCDA with multiple global weights
                    mcda.run_MCDA(file_path=None)
                    glob_dct[run][n] = winning_chance = \
                        mcda.winners[mcda.winners.Winner==alt].shape[0]/weight_num
                if winning_chance >= target_chance:
                    break
                n += 1
        glob_df = pd.DataFrame.from_dict(glob_dct).transpose()
        steps = glob_df.index.to_list()
        steps[0] = [steps[0]]*len(steps[1])
        steps = np.array(steps)
        num_steps = steps.shape[1]
        for i in range(num_steps):
            glob_df.insert(0, f'step{num_steps-i}', steps[:,-i])
        return glob_dct, glob_df

    # Firstly run all the combinations
    cutoff_step = cutoff_step or len(inds) # exhaust all indicators if `cutoff_step` not provided
    runs = list(combinations(inds, cutoff_step))
    glob_dct_comb, glob_df_comb = iter_run(runs)

    # Select all iterations that can reach the maximum
    max_iter = glob_df_comb[glob_df_comb.iloc[:, -1]==glob_df_comb.max().iloc[-1]].index.to_list()
    max_inds = set(sum([i for i in max_iter], ())) # use set to eliminate repetitive ones

    # Update the best scores (ones that are not from a manual scale,
    # but from the min/max of all alternatives)
    for k, v in best_score_dct.items():
        if v == None:
            best_score_dct[k] = oat_dct[k]['updated']
    # Add in indicators where the other alternatives have already reached the best score
    temp_scores = baseline_indicator_scores.copy()
    best_df = pd.DataFrame.from_dict(best_score_dct, orient='index')
    temp_scores = temp_scores[best_df.index] # exclude indicators without best scores
    temp_scores.loc[alt_idx] = best_df.values.T[0]
    add_inds = temp_scores.transpose()[temp_scores.apply(pd.Series.nunique)==1].index.to_list()
    perm_inds = max_inds.union(set(add_inds))
    perm_inds = perm_inds.intersection(set(inds)).difference(set(already_best))

    # Run permutations from the indicators that can either reach the target winning chance
    # within the cutoff_step, or will lead to the elimination of the indicator in local weighing
    # (e.g., where the alternative of interest will catch up with the other alternatives
    # and result in all alternatives having the same indicator score)
    perm_runs = list(permutations(perm_inds, cutoff_step))
    glob_dct_perm, glob_df_perm = iter_run(perm_runs)

    save_pickle(glob_dct_comb, os.path.join(results_path, f'improvements/glob_dct_comb_{wt_sce_num}.pckl'))
    save_pickle(glob_dct_perm, os.path.join(results_path, f'improvements/glob_dct_perm_{wt_sce_num}.pckl'))

    if file_path is not None:
        file_path = file_path if file_path != '' \
            else os.path.join(results_path, f'improvements/global_optimum_{wt_sce_num}.xlsx')
        writer = pd.ExcelWriter(file_path)
        glob_df_comb.to_excel(writer, sheet_name='combinations')
        glob_df_perm.to_excel(writer, sheet_name='permutations')
        writer.save()

    return glob_dct_comb, glob_df_comb, glob_dct_perm, glob_df_perm


# %%

# =============================================================================
# Plot the global trajectories
# =============================================================================

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

def plot_global_trajectory(glob_dct, wt_sce_num, file_path=''):
    fig, ax = plt.subplots(figsize=(6, 8))
    baseline_x, baseline_y = zip(*glob_dct.pop('baseline').items())
    for idx, chance_dct in glob_dct.items():
        x = (*baseline_x, *chance_dct.keys())
        y = (*baseline_y, *chance_dct.values())

        # Plot the line graph in segments with different colors
        c = get_colors(idx)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=c)
        lc.set_array(x[1:])
        lc.set_linewidth(0.5)
        ax.add_collection(lc)
    glob_dct['baseline'] = dict.fromkeys(baseline_x, *baseline_y) # add back the baseline

    ax.set(xticks=x, xlabel='Number of changed indicators',
           yticks=[0, 0.2, 0.4, 0.6, 0.8, 1], ylabel='Winning chance')
    ax.grid(axis='x', color=Guest_colors.gray.RGBn, linestyle='--', linewidth=0.5)

    if file_path is not None:
        file_path = file_path if file_path != '' \
            else os.path.join(figures_path, f'improvements_global_{wt_sce_num}.png')
        ax.figure.savefig(file_path, dpi=300)
    return ax


def plot_global_success(glob_dct_perm, wt_sce_num, file_path=''):
    # Only plot ones that can reach 100% winning chance
    glob_dct_success = glob_dct_perm.copy()
    temp_dct = glob_dct_success.copy()
    for idx, chance_dct in temp_dct.items():
        if idx == 'baseline':
            continue
        else:
            chances = list(chance_dct.values())
            if chances[-1] != 1:
                glob_dct_success.pop(idx)
    file_path = file_path or \
        os.path.join(figures_path, f'improvements_global_success_{wt_sce_num}.png')
    ax_success = plot_global_trajectory(
        glob_dct_success, wt_sce_num=sce_num2,
        file_path=file_path)
    return ax_success



# %%

# =============================================================================
# For each indicator, find the ranges of winning chances with it and without it
# =============================================================================

def get_indicator_chances(glob_df_comb, file_path=''):
    df = glob_df_comb.copy()
    # Take care of the baseline
    baseline = df.loc['baseline'].to_list()
    [baseline] = [i for i in baseline if not str(i).isalpha()]
    df.drop('baseline', inplace=True)
    # Get all indicators
    num_step = df.shape[1]//2 # one extra column for baseline
    inds = df.iloc[:, :num_step].stack().unique().tolist()
    inds.sort()
    categorize_ind = lambda criterion: [i for i in inds if i.startswith(criterion)]
    inds = categorize_ind('T')+categorize_ind('RR')+categorize_ind('Env') \
        +categorize_ind('Econ')+categorize_ind('S')

    # Differentiate the results into those including a particular indicator
    # and those excluding this one
    final_chance = df[num_step]
    idxs = final_chance.index
    include_df, exclude_df = pd.DataFrame(), pd.DataFrame()
    for ind in inds: # iterate through all indicators
        include, exclude = [], []
        for idx in idxs: # iterate through all indices (each index is one combination)
            if ind in idx:
                include.append(idx)
            else:
                exclude.append(idx)
        include_df[ind] = final_chance[include].values
        exclude_df[ind] = final_chance[exclude].values

    if file_path is not None:
        file_path = file_path if file_path != '' \
            else os.path.join(results_path, 'improvements/indicator_chances.xlsx')
        writer = pd.ExcelWriter(file_path)
        include_df.to_excel(writer, sheet_name='include')
        exclude_df.to_excel(writer, sheet_name='exclude')
        writer.save()

    return include_df, exclude_df


def plot_indicator_chances(indicator_df, file_path=''):
    sns.set_theme(style='white', rc={'axes.facecolor': (0, 0, 0, 0)})
    df = indicator_df.stack().reset_index().drop('level_0', axis=1)
    df = df.rename(columns={'level_1': 'indicator', 0: 'chance'})
    df.insert(0, 'criterion', df['indicator'].apply(lambda x:x[:-1]))

    # Make palette
    pal = dict.fromkeys(df.indicator.unique())
    for k in pal.keys():
        pal[k] = color_dct[k[:-1]]

    # Initialize the FacetGrid object
    g = sns.FacetGrid(df, row='indicator', hue='indicator', aspect=15, height=.5,
                      palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, 'chance',
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, 'chance', clip_on=False, color='w', lw=2, bw_adjust=.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle='-', color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight='bold', color=color,
                ha='left', va='center', transform=ax.transAxes)
    g.map(label, 'chance')

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles('')
    g.set(yticks=[], ylabel='')
    g.despine(bottom=True, left=True)

    # Other fine tuning
    last_ax = g.axes[-1].item()
    last_ax.set(xlabel='Winning chance')

    if file_path is not None:
        file_path = file_path if file_path != '' \
            else os.path.join(figures_path, 'improvements_indicator_chance.png')
        g.figure.savefig(file_path, dpi=300)
    return g


# %%

if __name__ == '__main__':
    # Using original number of criterion weight scenarios
    file_path = os.path.join(results_path, f'criterion_weights_{sce_num1}.xlsx')
    weight_df1 = pd.read_excel(file_path)
    bwaise_mcda.criterion_weights = weight_df1

    # One-at-a-time at baseline
    oat_dct = test_oat(bwaise_mcda, alt='Alternative C', best_score=best_score_dct)
    ax_oat = plot_oat(oat_dct, wt_sce_num=sce_num1)

    # # Local optimum
    # loc_dct, loc_df = local_optimum_approach(
    #     bwaise_mcda, alt='Alternative C', oat_dct=oat_dct, wt_sce_num=sce_num1)
    # ax_loc = plot_local_optimum(loc_dct, wt_sce_num=sce_num1)

    # # Smaller number of criterion weight scenarios
    sce_num2 = 100 # use fewer scenarios here
    # weight_df2 = bwaise_mcda.generate_weights(criterion_num=criterion_num, wt_scenario_num=sce_num2)
    # bwaise_mcda.criterion_weights = weight_df2
    # file_path = os.path.join(results_path, f'criterion_weights_{sce_num2}.xlsx')
    # weight_df2.to_excel(file_path, sheet_name='Criterion weights')
    # # One-at-a-time at baseline
    # oat_dct = test_oat(bwaise_mcda, alt='Alternative C', best_score=best_score_dct)
    # ax_oat2 = plot_oat(oat_dct, wt_sce_num=sce_num2)
    # # Local optimum
    # loc_dct, loc_df = local_optimum_approach(
    #     bwaise_mcda, alt='Alternative C', oat_dct=oat_dct, wt_sce_num=sce_num2, file_path='')
    # ax_loc2 = plot_local_optimum(loc_dct, wt_sce_num=sce_num2)

    # # Global optimum
    # glob_dct_comb, glob_df_comb, glob_dct_perm, glob_df_perm = global_optimum_approach(
    #     bwaise_mcda, 'Alternative C', oat_dct, wt_sce_num=sce_num2,
    #     select_top=None, target_chance=1, cutoff_step=len(loc_dct)-1) # subtract 1 for baseline
    # # Trajectory plots
    # ax_comb = plot_global_trajectory(
    #     glob_dct_comb, wt_sce_num=sce_num2,
    #     file_path=os.path.join(figures_path, f'improvements_global_comb_{sce_num2}.png'))
    # ax_perm = plot_global_trajectory(
    #     glob_dct_perm, wt_sce_num=sce_num2,
    #     file_path=os.path.join(figures_path, f'improvements_global_perm_{sce_num2}.png'))
    # ax_success = plot_global_success(glob_dct_perm, sce_num2)
    # # Ridge plots (overlapping density plots)
    # include_df, exclude_df = get_indicator_chances(glob_df_comb)
    # g_include = plot_indicator_chances(
    #     include_df, file_path=os.path.join(figures_path, 'improvements_indicator_chance_include.png'))
    # g_exclude = plot_indicator_chances(
    #     exclude_df, file_path=os.path.join(figures_path, 'improvements_indicator_chance_exclude.png'))


# %%

# =============================================================================
# Use matplotlib for plotting, doesn't look very good
# =============================================================================

# def plot_indicator_chances(df, file_path=''):
#     fig, ax = plt.subplots(figsize=(6, 8))
#     ax.boxplot(df, vert=False)
#     ax.set(yticklabels=include_df.columns.to_list(),
#            xlabel='Winning chance')
#     if file_path is not None:
#         file_path = file_path if file_path != '' \
#             else os.path.join(figures_path, 'improvements_indicator_chance.png')
#         ax.figure.savefig(file_path, dpi=300)
#     return ax


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
#     mcda.indicator_scores = baseline_indicator_scores.copy()
#     idx = mcda.alt_names[mcda.alt_names==alt].index[0]

#     min_val = min_val if min_val else 0
#     max_val = max_val if max_val else baseline_indicator_scores.loc[idx, indicator]

#     # Total number of global weights
#     weight_num = mcda.criterion_weights.shape[0]

#     vals = np.linspace(min_val, max_val, num=step_num)
#     win_dct = {}
#     if not include_simulation_uncertainties: # fix other tech scores at the baseline

#         for val in vals:
#             mcda.indicator_scores.loc[idx, indicator] = val
#             mcda.run_MCDA(file_path=None)
#             win_dct[val] = mcda.winners[mcda.winners.Winner==alt].shape[0]/weight_num

#     else:
#         # Make a copy for manipulating
#         ind_score_dct_updated = ind_score_dct.copy()
#         for val in vals:
#             # Update all the costs to be the set value
#             for v in ind_score_dct_updated.values():
#                 ind_score_dct_updated.loc[idx, indicator] = val
#             score_df_dct, rank_df_dct, winner_df = \
#                 mcda.run_MCDA_multi_scores(ind_score_dct=ind_score_dct_updated)
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
#     ax0.figure.savefig(os.path.join(figures_path, 'Cwin_across_cost.png'), dpi=300)

#     # **Each** step takes ~1 hour (at each step, we are running 1000 simulation*1000 global weights)
#     Cwin_across_cost_uncertainty = test_across_axis(bwaise_mcda, alt='Alternative C',
#                                                     indicator='Econ1',
#                                                     include_simulation_uncertainties=True,
#                                                     min_val=0, step_num=2)
#     ax1 = plot_across_axis(Cwin_across_cost_uncertainty, include_simulation_uncertainties=True)
#     ax1.figure.savefig(os.path.join(figures_path, 'Cwin_across_cost_with_band.png'), dpi=300)