#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making of sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/DMsan/blob/main/LICENSE.txt
for license details.2

Run this module (after generating the data) to plot the figures.
'''

import os, numpy as np, pandas as pd, matplotlib as mpl
from matplotlib import pyplot as plt
from qsdsan.utils import load_pickle
from dmsan.comparison import results_path, figures_path

# Plot configuration
mpl.rcParams['font.sans-serif'] = 'arial'
mpl.rcParams["figure.autolayout"] = True


# %%

def make_boxplots(data, save=True):
    ax_dct = {}
    for country, country_data in data.items():
        uncertainty_data = country_data['uncertainty']
        score_df_dct = uncertainty_data['score_df_dct'] # keys are the global weight scenarios
        # Compile dataframe of all weight scenarios and uncertainty in indicator scores
        compiled_df = pd.concat(score_df_dct.values())
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.boxplot(
            compiled_df,
            sym="", # hide the fliers (outliers)
            whis=(5, 95), # 5-95 percentiles for whiskers, unfortunately edges cannot be adjusted
            )
        ax.set(
            xticklabels=compiled_df.columns,
            ylabel='Performance score',
            )
        ax_dct[country] = ax
        if save: fig.savefig(os.path.join(figures_path, f'boxplot_{country}.png'))
    return ax_dct


# %%

def format_linegraph_ax(ax, title=''):
    ax.set(title=title,
           xlim=(0, 4), ylim=(0, 1), ylabel='Criterion Weights',
           xticks=(0, 1, 2, 3, 4),
           xticklabels=('T', 'RR', 'Env', 'Econ', 'S'))
    return ax

def make_linegraphs(data, save=True):
    # To extract the weighing information
    # def ratio2float(ratio):
    #     try: return np.array(ratio.strip("'").strip('"').split(':'), dtype='float')
    #     except: breakpoint()
    ratio2float = lambda ratio: np.array(ratio.strip("'").strip('"').split(':'), dtype='float')

    ax_dct = {}
    for country, country_data in data.items():
        uncertainty_data = country_data['uncertainty']
        winner_dfT = uncertainty_data['winner_df'].T.copy() # columns are the global weight scenarios
        modules = uncertainty_data['spearman_dct'].keys()
        for module in modules:
            winner_dfT[module] = winner_dfT[winner_dfT==module].count(axis=1)

        tally = winner_dfT.iloc[:, -len(modules):]
        weight_winner = tally.idxmax(axis=1)

        # Color for each module
        colors = (
            (0.635, 0.502, 0.725, 1), # the last one is alpha (1 means no transparency)
            (0.635, 0.502, 0.725, 1),
            (0.376, 0.757, 0.812, 1),
            (0.376, 0.757, 0.812, 1),
            (0.929, 0.345, 0.435, 1),
            (0.929, 0.345, 0.435, 1),
            )
        for module, color in zip(modules, colors):
            ax_dct[module] = module_ax_dct = {}
            fig, ax = plt.subplots(figsize=(8, 4.5))
            module_won = (weight_winner[weight_winner==module]).index
            for wt in module_won:
                #!!! Need to figure out the indexing problem,
                # possibily because of bugs in uncertainty_sensitiviy.py
                try: ax.plot(ratio2float(wt), color=color, linewidth=0.5)
                except: breakpoint()
            ax = format_linegraph_ax(ax, title=module)
            module_ax_dct[module] = ax
            if save: fig.savefig(os.path.join(figures_path, f'linegraph_{country}_{module}.png'))
    return ax_dct



# %%

if __name__ == '__main__':
    # Load the cached data
    data = load_pickle(os.path.join(results_path, 'data.pckl'))
    box_ax_dct = make_boxplots(data, save=True)
    linegraph_ax_dct = make_linegraphs(data, save=True)