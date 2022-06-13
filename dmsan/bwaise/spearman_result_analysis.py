# -*- coding: utf-8 -*-
"""
@author:
    Joy Zhang <joycheung1994@gmail.com>

Run this script to analyze the results from Spearman correlation analysis.
"""

import os, pandas as pd, matplotlib.pyplot as plt
from dmsan.bwaise import scores_path, results_path, figures_path
folder = os.path.join(results_path, 'sensitivity')
file = "performance_Spearman_scores.pckl"

idx = pd.IndexSlice
param_path = os.path.join(scores_path, 'parameters_annotated.xlsx')
pars = pd.read_excel(param_path, sheet_name=None)
def parse_tuple(st):
    return tuple([i.strip('\(\'\)') for i in st.split(r', ')])

for alt, df in pars.items():
    df['Parameters'] = df.Parameters.apply(parse_tuple)

def plot_spearman(corr_dct):
    for alt, df in corr_dct.items():
        rhos = df.xs('rho', axis=1, level='Stats').T
        rhos.columns = pars[f'Alternative {alt}']['Parameters']
        median = rhos.median()
        median.sort_values(ascending=True, inplace=True)
        rhos = rhos[median.index]
        fig, ax = plt.subplots(figsize=(18, 35))
        ax.boxplot(rhos, vert=False,
                   showfliers=False,
                   showmeans=True,
                   labels=rhos.columns)
        ax.set(xlabel='Spearman correlation coefficient (for MCDA scores)')
        name = f'Spearman_rho_{alt}.png'
        plt.tight_layout()
        fig.savefig(os.path.join(figures_path, name), 
                    dpi=300, 
                    bbox_inches = "tight")

if __name__ == '__main__':
    corr = pd.read_pickle(os.path.join(folder, file))
    plot_spearman(corr)
