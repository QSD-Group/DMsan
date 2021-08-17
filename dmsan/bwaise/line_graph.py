#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 08:28:41 2021

@author: Yalin Li <zoe.yalin.li@gmail.com>

Run this module to make line graphs with X-axis being the different criteria,
Y-axis being criterion weight, and line color representing the probability of
an alternative having the highest score among all.
"""

import os, pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, lines as mlines, pylab as pl
from qsdsan.utils.colors import palettes, Color
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

colors = [
    Color('gray', '#d0d1cd').RGBn,
    Color('blue', '#60c1cf').RGBn,
    Color('orange', '#fa8f61').RGBn,
    Color('dark_green', '#4d7a53').RGBn,
    ]

# Make line graphs using different colors for different cutoffs
def make_line_graph1(winner_df, alt, cutoffs=[0.25, 0.5, 0.75, 1],
                    colors=colors, include_legend=True):
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

    if include_legend:
        ax.legend(handles=handles)

    ax.set(title=alt,
            xlim=(0, 4), ylim=(0, 1), ylabel='Criteria Weights',
            xticks=(0, 1, 2, 3, 4),
            xticklabels=('T', 'RR', 'Env', 'Econ', 'S'))

    return fig, ax


# %%

# Make line graphs using gradient based on tutorials below
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
# https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
def make_line_graph2A(winner_df, alt, cmap):
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


# Only plot the winner
def make_line_graph2B(winner_df):
    # % of times that the select alternative wins
    tot = winner_df.shape[0]
    alts = ['A', 'B', 'C']
    # alts = [f'Alternative {alt}' for alt in ('A', 'B', 'C')]
    percents = pd.concat([winner_df[winner_df==f'Alternative {i}'].count()/tot
                          for i in alts],
                         axis=1)
    percents.columns = alts
    separated = [getattr(percents[percents.max(axis=1)==getattr(percents, i)], i)
                 for i in alts]

    # Extract the weighing information
    ration2float = lambda ratio: np.array(ratio.split(':'), dtype='float')
    wts = [np.array([ration2float(i) for i in wt.index]) for wt in separated]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for wt, cmap in zip(wts, ('Reds', 'Greens', 'Blues')):
        ax.plot(wt.transpose(), color=getattr(pl.cm, cmap)(225),
                linewidth=0.5)

    ax.set(title='Overall winner',
           xlim=(0, 4), ylim=(0, 1), ylabel='Criteria Weights',
           xticks=(0, 1, 2, 3, 4),
           xticklabels=('T', 'RR', 'Env', 'Econ', 'S'))

    return fig, ax


# %%

# =============================================================================
# Lazy codes to make figures
# =============================================================================

def make_line_graphs(save=True):
    fig1A, ax1A = make_line_graph1(winner_df, 'Alternative A', include_legend=False)
    fig1B, ax1B = make_line_graph1(winner_df, 'Alternative B', include_legend=False)
    fig1C, ax1C = make_line_graph1(winner_df, 'Alternative C', include_legend=False)
    fig2A, ax2A = make_line_graph2A(winner_df, 'Alternative A', 'Reds')
    fig2B, ax2B = make_line_graph2A(winner_df, 'Alternative B', 'Greens')
    fig2C, ax2C = make_line_graph2A(winner_df, 'Alternative C', 'Blues')
    fig2D, ax2D = make_line_graph2B(winner_df)

    if save:
        fig1A.savefig(os.path.join(figures_path, '1A.png'), dpi=100)
        fig1B.savefig(os.path.join(figures_path, '1B.png'), dpi=100)
        fig1C.savefig(os.path.join(figures_path, '1C.png'), dpi=100)
        fig2A.savefig(os.path.join(figures_path, '2A.png'), dpi=100)
        fig2B.savefig(os.path.join(figures_path, '2B.png'), dpi=100)
        fig2C.savefig(os.path.join(figures_path, '2C.png'), dpi=100)
        fig2D.savefig(os.path.join(figures_path, '2D.png'), dpi=100)


if __name__ == '__main__':
    make_line_graphs()