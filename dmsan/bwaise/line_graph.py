#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 08:28:41 2021

@author: Yalin Li
"""

import os, pickle
import numpy as np
from matplotlib import pyplot as plt, lines as mlines, pylab as pl
from matplotlib.colors import LinearSegmentedColormap as LSC
from qsdsan.utils import palettes
from dmsan.bwaise import results_path, figures_path

Guest = palettes['Guest']


# %%

# =============================================================================
# Loading pickle files
# =============================================================================

def import_pickle(baseline=True, uncertainty=True, sensitivity='KS'):
    def load(path):
        f = open(path, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj

    loaded = dict.fromkeys(('baseline', 'uncertainty', 'sensitivity'))

    if baseline:
        file_path = os.path.join(results_path, 'bwaise_mcda.pckl')
        loaded['baseline'] = load(file_path)

    if uncertainty:
        file_path = os.path.join(results_path, 'uncertainty/AHP_TOPSIS.pckl')
        loaded['uncertainty'] = load(file_path)

    if sensitivity:
        file_path = os.path.join(results_path, f'sensitivity/AHP_TOPSIS_{sensitivity}_ranks.pckl')
        loaded['sensitivity'] = [load(file_path)]

        if sensitivity != 'KS':
            file_path = os.path.join(results_path, f'sensitivity/AHP_TOPSIS_{sensitivity}_scores.xlsx')
            loaded['sensitivity'].append(load(file_path))

    return loaded

loaded = import_pickle(baseline=True, uncertainty=True, sensitivity='KS')
bwaise_mcda = loaded['baseline']
score_df_dct, rank_df_dct, winner_df = loaded['uncertainty']
rank_corr_dct = loaded['sensitivity']


# %%

# =============================================================================
# Make line graphs
# =============================================================================

colors = [Guest.gray.RGBn, Guest.blue.RGBn, Guest.red.RGBn, Guest.green.RGBn]

# Make line graphs using different colors for different cutoffs
def make_line_graph1(winner_df, alt, cutoffs=[0.25, 0.5, 0.75, 1],
                    colors=colors):
    if len(cutoffs) != len(colors):
        raise ValueError(f'The number of `cutoffs` ({len(cutoffs)}) '
                          f'should equal the number of `colors` ({len(colors)}).')

    # % of times that the select alternative wins
    percent = winner_df[winner_df==alt].count()/winner_df.shape[0]

    # Extract the weighing information
    ration2float = lambda ratio: np.array(ratio.split(':'), dtype='float')

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Separate into the over and under criterion groups and make the corresponding lines
    handles = []
    for idx in range(len(cutoffs)):
        lower = 0. if idx == 0 else cutoffs[idx-1]
        upper = cutoffs[idx]
        if upper == 1:
            wt = np.array([ration2float(i)
                            for i in percent[(lower<=percent)&(percent<=upper)].index])
            right = ']'
        else:
            wt = np.array([ration2float(i)
                            for i in percent[(lower<=percent)&(percent<upper)].index])
            right = ')'

        if wt.size == 0:
            continue

        # Transpose the shape into that needed for plotting
        # (# of the weighing aspects (e.g., technical vs. economic), # of over/under the criterion),
        # when used in plotting the line graph, each row will be the y-axis value of the line
        # (x-axis value represents the N criteria)
        ax.plot(wt.transpose(), color=colors[idx], linewidth=0.5)
        handles.append(mlines.Line2D([], [], color=colors[idx],
                                      label=f'[{lower:.0%}, {upper:.0%}{right}'))

    ax.legend(handles=handles)
    ax.set(title=alt,
            xlim=(0, 4), ylim=(0, 1), ylabel='Criteria Weights',
            xticks=(0, 1, 2, 3, 4),
            xticklabels=('T', 'RR', 'Env', 'Econ', 'S'))

    return fig, ax

figA1, axA1 = make_line_graph1(winner_df, 'Alternative A')
figB1, axB1 = make_line_graph1(winner_df, 'Alternative B')
figC1, axC1 = make_line_graph1(winner_df, 'Alternative C')

figA1.savefig(os.path.join(figures_path, 'A1.png'), dpi=100)
figB1.savefig(os.path.join(figures_path, 'B1.png'), dpi=100)
figC1.savefig(os.path.join(figures_path, 'C1.png'), dpi=100)


# %%

# Make line graphs using the same color, but different transparency
def make_line_graph2(winner_df, alt, color='b'):
    # % of times that the select alternative wins
    percent = winner_df[winner_df==alt].count()/winner_df.shape[0]

    # Extract the weighing information
    ration2float = lambda ratio: np.array(ratio.split(':'), dtype='float')
    wt = np.array([ration2float(i) for i in percent.index])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(wt.shape[0]):
        ax.plot(wt[i], color=color, linewidth=0.5, alpha=percent[i])

    ax.set(title=alt,
            xlim=(0, 4), ylim=(0, 1), ylabel='Criteria Weights',
            xticks=(0, 1, 2, 3, 4),
            xticklabels=('T', 'RR', 'Env', 'Econ', 'S'))

    return fig, ax

figA2, axA2 = make_line_graph2(winner_df, 'Alternative A')
figB2, axB2 = make_line_graph2(winner_df, 'Alternative B')
figC2, axC2 = make_line_graph2(winner_df, 'Alternative C')

figA2.savefig(os.path.join(figures_path, 'A2.png'), dpi=100)
figB2.savefig(os.path.join(figures_path, 'B2.png'), dpi=100)
figC2.savefig(os.path.join(figures_path, 'C2.png'), dpi=100)


# %%

# Make line graphs using gradient based on tutorials below
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
# https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
def make_line_graph3(winner_df, alt, cmap):
    # % of times that the select alternative wins
    percent = winner_df[winner_df==alt].count()/winner_df.shape[0]

    # Extract the weighing information
    ration2float = lambda ratio: np.array(ratio.split(':'), dtype='float')
    wt = np.array([ration2float(i) for i in percent.index])
    fig, ax = plt.subplots(figsize=(8, 4.5))

    cm = getattr(pl.cm, cmap)
    for i in range(wt.shape[0]):
        ax.plot(wt[i], color=cm(percent[::1])[i],
                linewidth=0.5, alpha=percent[i])

    ax.set(title=alt,
           xlim=(0, 4), ylim=(0, 1), ylabel='Criteria Weights',
           xticks=(0, 1, 2, 3, 4),
           xticklabels=('T', 'RR', 'Env', 'Econ', 'S'))

    return fig, ax

figA3, axA3 = make_line_graph3(winner_df, 'Alternative A', 'Reds')
figB3, axB3 = make_line_graph3(winner_df, 'Alternative B', 'Greens')
figC3, axC3 = make_line_graph3(winner_df, 'Alternative C', 'Blues')

figA3.savefig(os.path.join(figures_path, 'A3.png'), dpi=100)
figB3.savefig(os.path.join(figures_path, 'B3.png'), dpi=100)
figC3.savefig(os.path.join(figures_path, 'C3.png'), dpi=100)


# %%

# Make line graphs using gradient colors
def make_line_graph4(winner_df, alt, cmap):
    # % of times that the select alternative wins
    percent = winner_df[winner_df==alt].count()/winner_df.shape[0]

    # Extract the weighing information
    ration2float = lambda ratio: np.array(ratio.split(':'), dtype='float')
    wt = np.array([ration2float(i) for i in percent.index])
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i in range(wt.shape[0]):
        ax.plot(wt[i], color=cmap(percent[::1])[i],
                linewidth=0.5, alpha=percent[i])

    ax.set(title=alt,
           xlim=(0, 4), ylim=(0, 1), ylabel='Criteria Weights',
           xticks=(0, 1, 2, 3, 4),
           xticklabels=('T', 'RR', 'Env', 'Econ', 'S'))

    return fig, ax


colorlist = [Guest.red.RGBn, Guest.green.RGBn]
cmap = LSC.from_list('temp', colorlist)

figA4, axA4 = make_line_graph4(winner_df, 'Alternative A', cmap)
figB4, axB4 = make_line_graph4(winner_df, 'Alternative B', cmap)
figC4, axC4 = make_line_graph4(winner_df, 'Alternative C', cmap)

figA4.savefig(os.path.join(figures_path, 'A4.png'), dpi=100)
figB4.savefig(os.path.join(figures_path, 'B4.png'), dpi=100)
figC4.savefig(os.path.join(figures_path, 'C4.png'), dpi=100)