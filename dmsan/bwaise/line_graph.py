#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yalin Li <mailto.yalin.li@gmail.com>

Run this module to make line graphs with X-axis being the different criteria,
Y-axis being criterion weight, and line color representing the probability of
an alternative having the highest score among all.
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, pylab as pl
from dmsan.bwaise import figures_path, import_from_pickle

loaded = import_from_pickle(uncertainty=True, sensitivity='KS')
score_df_dct, rank_df_dct, winner_df = loaded['uncertainty']
rank_corr_dct = loaded['sensitivity']


# %%

def format_ax(ax, title=''):
    ax.set(title=title,
           xlim=(0, 4), ylim=(0, 1), ylabel='Criterion Weights',
           xticks=(0, 1, 2, 3, 4),
           xticklabels=('T', 'RR', 'Env', 'Econ', 'S'))
    return ax


# Plot the winner in one figure
def plot_winner(winner_df):
    # % of scenarios that the select alternative wins
    tot = winner_df.shape[1]
    alts = ['A', 'B', 'C']
    percents = pd.concat([winner_df[winner_df==f'Alternative {i}'].count()/tot
                          for i in alts],
                          axis=1)
    percents.columns = alts
    separated = [getattr(percents[percents.max(axis=1)==getattr(percents, i)], i)
                  for i in alts]
    counts = [i.size for i in separated]
    for n, alt in enumerate(alts):
        print(f'Alternative {alt} wins {counts[n]} of {tot} times.')
    # Extract the weighing information
    ratio2float = lambda ratio: np.array(ratio.split(':'), dtype='float')
    wts = [np.array([ratio2float(i) for i in wt.index]) for wt in separated]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for wt, cmap in zip(wts, ('Reds', 'Greens', 'Blues')):
        ax.plot(wt.transpose(), color=getattr(pl.cm, cmap)(225),
                linewidth=0.5)
    ax = format_ax(ax, title='Overall winner')
    return fig, ax


def plot_solid_color(winner_df):
    # % of scenarios that the select alternative wins
    tot = winner_df.shape[1]
    alts = ['A', 'B', 'C']
    percents = pd.concat(
        [winner_df[winner_df==f'Alternative {i}'].count()/tot for i in alts],
        axis=1)
    percents.columns = alts
    separated = [getattr(percents[percents.max(axis=1)==getattr(percents, i)], i)
                 for i in alts]

    # Extract the weighing information
    ratio2float = lambda ratio: np.array(ratio.split(':'), dtype='float')
    wts = [np.array([ratio2float(i) for i in wt.index]) for wt in separated]

    figs = []
    axes = []
    for alt, wt, cmap in zip(alts, wts, ('Reds', 'Greens', 'Blues')):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(wt.transpose(), color=getattr(pl.cm, cmap)(225),
                linewidth=0.5)
        ax = format_ax(ax, title=f'Alternative {alt}')
        figs.append(fig)
        axes.append(ax)

    return figs, axes


# Make line graphs using gradient based on tutorials below
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
# https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
def plot_gradient_color(winner_df, alt, cmap):
    # % of times that the select alternative wins
    percent = winner_df[winner_df==alt].count()/winner_df.shape[0]

    # Extract the weighing information
    ratio2float = lambda ratio: np.array(ratio.split(':'), dtype='float')
    wt = np.array([ratio2float(i) for i in percent.index])
    fig, ax = plt.subplots(figsize=(8, 4.5))

    cm = getattr(pl.cm, cmap)
    for i in range(wt.shape[0]):
        ax.plot(wt[i], color=cm(percent[::1])[i],
                linewidth=0.5, alpha=percent[i])
    ax = format_ax(ax, title=alt)
    return fig, ax


# %%

# =============================================================================
# Make figures
# =============================================================================

def make_line_graphs(save=True):
    fig_winner, ax_winner = plot_winner(winner_df)
    fig_gradient_A, ax_gradient_A = plot_gradient_color(winner_df, 'Alternative A', 'Reds')
    fig_gradient_B, ax_gradient_B = plot_gradient_color(winner_df, 'Alternative B', 'Greens')
    fig_gradient_C, ax_gradient_C = plot_gradient_color(winner_df, 'Alternative C', 'Blues')
    figs, axes = plot_solid_color(winner_df)
    fig_solid_A, fig_solid_B, fig_solid_C = figs

    if save:
        fig_winner.savefig(os.path.join(figures_path, 'winner.png'), dpi=300)
        fig_solid_A.savefig(os.path.join(figures_path, 'A_solid.png'), dpi=300)
        fig_solid_B.savefig(os.path.join(figures_path, 'B_solid.png'), dpi=300)
        fig_solid_C.savefig(os.path.join(figures_path, 'C_solid.png'), dpi=300)
        fig_gradient_A.savefig(os.path.join(figures_path, 'A_gradient.png'), dpi=300)
        fig_gradient_B.savefig(os.path.join(figures_path, 'B_gradient.png'), dpi=300)
        fig_gradient_C.savefig(os.path.join(figures_path, 'C_gradient.png'), dpi=300)

if __name__ == '__main__':
    make_line_graphs(save=True)


# %%

# # =============================================================================
# # Legacy code to use different colors for winning chance below different cutoffs
# # =============================================================================

# from matplotlib import lines as mlines
# from qsdsan.utils.colors import Color

# colors = [
#     Color('gray', '#d0d1cd').RGBn,
#     Color('blue', '#60c1cf').RGBn,
#     Color('orange', '#fa8f61').RGBn,
#     Color('dark_green', '#4d7a53').RGBn,
#     ]

# def line_graph_cutoff(winner_df, alt, cutoffs=[0.25, 0.5, 0.75, 1],
#                       colors=colors, include_legend=True):
#     if len(cutoffs) != len(colors):
#         raise ValueError(f'The number of `cutoffs` ({len(cutoffs)}) '
#                           f'should equal the number of `colors` ({len(colors)}).')

#     # % of times that the select alternative wins
#     percent = winner_df[winner_df==alt].count()/winner_df.shape[0]

#     # Extract the weighing information
#     ratio2float = lambda ratio: np.array(ratio.split(':'), dtype='float')

#     fig, ax = plt.subplots(figsize=(8, 4.5))

#     # Separate into the over and under criterion groups and make the corresponding lines
#     handles = []
#     for idx in range(len(cutoffs)):
#         lower = 0. if idx == 0 else cutoffs[idx-1]
#         upper = cutoffs[idx]
#         if upper == 1:
#             wt = np.array([ratio2float(i)
#                             for i in percent[(lower<=percent)&(percent<=upper)].index])
#             right = ']'
#         else:
#             wt = np.array([ratio2float(i)
#                             for i in percent[(lower<=percent)&(percent<upper)].index])
#             right = ')'

#         if wt.size == 0:
#             continue

#         # Transpose the shape into that needed for plotting, the shape is
#         # (number of the weighing aspects (e.g., technical vs. economic), number of over/under the criterion),
#         # when used in plotting the line graph, each row will be the y-axis value of the line
#         # (x-axis value represents the N criteria)
#         ax.plot(wt.transpose(), color=colors[idx], linewidth=0.5)
#         handles.append(mlines.Line2D([], [], color=colors[idx],
#                                       label=f'[{lower:.0%}, {upper:.0%}{right}'))

#     if include_legend:
#         ax.legend(handles=handles)

#     ax.set(title=alt,
#             xlim=(0, 4), ylim=(0, 1), ylabel='Criteria Weights',
#             xticks=(0, 1, 2, 3, 4),
#             xticklabels=('T', 'RR', 'Env', 'Econ', 'S'))

#     return fig, ax