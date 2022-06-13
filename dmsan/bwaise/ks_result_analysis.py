# -*- coding: utf-8 -*-
"""
@author:
    Joy Zhang <joycheung1994@gmail.com>

Run this script to analyze the results from Kolmogorov–Smirnov test.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from dmsan.bwaise import scores_path, results_path, figures_path
folder = os.path.join(results_path, 'sensitivity')
file = "performance_KS_ranks.pckl"

idx = pd.IndexSlice
param_path = os.path.join(scores_path, 'parameters_annotated.xlsx')
pars = pd.read_excel(param_path, sheet_name=None)
def parse_tuple(st):
    return tuple([i.strip('\(\'\)') for i in st.split(r', ')])

for alt, df in pars.items():
    df['Parameters'] = df.Parameters.apply(parse_tuple)

def descriptive(ks_dct):
    out_1 = {}
    out_2 = {}
    for alt, df in ks_dct.items():
        df.dropna(axis=1, how='all', inplace=True)
        _D = df.loc[:, idx[:, 'D']].droplevel('Stats', axis=1)
        _p = df.loc[:, idx[:, 'p-value']].droplevel('Stats', axis=1)
        _D_sig = (_D * (_p <= 0.05).astype('int')).replace(0, np.nan)

        out = pd.DataFrame()
        out['Parameters'] = df.loc[:, ('', 'Parameter')]
        out['percent_significant'] = _p.apply(lambda x: sum(x <= 0.05)/len(x)*100, axis=1)
        out['mean_signf_D'] = _D_sig.mean(axis=1, skipna=True)
        out['median_signf_D'] = _D_sig.median(axis=1, skipna=True)
        out['5th_pct'] = _D_sig.quantile(q=0.05, axis=1)
        out['95th_pct'] = _D_sig.quantile(q=0.95, axis=1)
        out['left_err'] = out.mean_signf_D - out['5th_pct']
        out['right_err'] = out['95th_pct'] - out.mean_signf_D
        out = pd.merge(out, pars[f'Alternative {alt}'].loc[:, 'Parameters':'S'], on='Parameters')
        out['No._of_criteria'] = out.loc[:, 'T':'S'].sum(axis=1)
        out_1[alt] = out

        _D_sig.index = df.loc[:, ('', 'Parameter')]
        _D_rank = _D_sig.T.dropna(axis=1, how='all')
        ncol = max(_D_rank.count(axis=1))
        nrow = _D_rank.shape[0]
        out = pd.DataFrame(index=_D_rank.index, columns=range(ncol))
        for i in range(nrow):
            srted = _D_rank.iloc[i, :].sort_values(ascending=False)
            n = srted.count()
            out.iloc[i, :n] = srted.index[:n]
        out_2[alt] = out
    return out_1, out_2

def export_to_excel(file_path, dct):
    with pd.ExcelWriter(file_path) as writer:
        for k, v in dct.items():
            v.to_excel(writer, sheet_name=k)

def make_scatter(dct, path):
    for alt, df in dct.items():
        fig, ax = plt.subplots(figsize=(8,6))
        size1 = np.ma.masked_where(df.DV, df['No._of_criteria']*20+20)
        size2 = np.ma.masked_where(1-df.DV, df['No._of_criteria']*20+20)
        color = ['#90918E' if i == 0 else '#79BF82' if i == 1 else '#60c1cf' if i == 2 else '#A280B9' if i == 3 else '#ED586F' for i in df['No._of_criteria']]
        ax.errorbar(x=df.percent_significant,
                    y=df.mean_signf_D,
                    yerr=[df.left_err, df.right_err],
                    fmt='none',
                    elinewidth=0.5,
                    capsize=1,
                    ecolor='grey',
                    alpha=0.8)
        ax.scatter(x=df.percent_significant,
                   y=df.mean_signf_D,
                   s=size1,
                   # c=np.sqrt(df['No._of_criteria']),
                   c=color,
                   marker='x',
                   alpha=1)
        ax.scatter(x=df.percent_significant,
                   y=df.mean_signf_D,
                   s=size2,
                   # c=np.sqrt(df['No._of_criteria']),
                   c=color,
                   marker='^',
                   alpha=1)
        ax.set(xlim=(0, 100), ylim=(0,1),
               xlabel='Percent significant',
               ylabel='mean D value of significant samples')

        name = f'ks_scatter_{alt}.png'
        fig.savefig(os.path.join(path, name), dpi=300)


def analyze(folder, path):
    ks = pd.read_pickle(os.path.join(folder, path))
    dct1, dct2 = descriptive(ks)
    path1 = os.path.join(folder, 'KS_descriptive_stats.xlsx')
    path2 = os.path.join(folder, 'KS_weights_vs_topX.xlsx')
    export_to_excel(path1, dct1)
    export_to_excel(path2, dct2)
    make_scatter(dct1, figures_path)

if __name__ == '__main__':
    analyze(folder, file)