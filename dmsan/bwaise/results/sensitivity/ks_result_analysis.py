# -*- coding: utf-8 -*-
"""
@author: 
    Joy Cheung <joycheung1994@gmail.com>
"""

import pandas as pd
import numpy as np
import os
# os.chdir("C:/Users/joy_c/Dropbox/PhD/Research/QSD/codes_developing/DMsan/dmsan/bwaise/results/sensitivity")
folder = os.path.dirname(__file__)
path = "AHP_TOPSIS_KS_ranks.pckl"

idx = pd.IndexSlice

def descriptive(ks_dct):
    out_dct = {}
    for alt, df in ks_dct.items():
        df.dropna(axis=1, how='all', inplace=True)
        _D = df.loc[:, idx[:, 'D']].droplevel('Stats', axis=1)
        _p = df.loc[:, idx[:, 'p-value']].droplevel('Stats', axis=1)
        _D_sig = (_D * (_p <= 0.05).astype('int')).replace(0, np.nan)
        out = pd.DataFrame()
        out['parameter'] = df.loc[:, ('', 'Parameter')]
        out['percent_significant'] = _p.apply(lambda x: sum(x <= 0.05)/len(x)*100, axis=1)
        out['mean_signf_D'] = _D_sig.mean(axis=1, skipna=True)
        out['median_signf_D'] = _D_sig.median(axis=1, skipna=True)
        out['5th_pct'] = _D_sig.quantile(q=0.05, axis=1)
        out['95th_pct'] = _D_sig.quantile(q=0.95, axis=1)
        out_dct[alt] = out
    return out_dct


def weight_vs_topx(ks_dct):
    out_dct = {}
    for alt, df in ks_dct.items():
        _D = df.loc[:, idx[:, 'D']].droplevel('Stats', axis=1)
        _D.index = df[('','Parameter')]
        out_dct[alt] = _D.astype(float).idxmax()
    return out_dct

def export_to_excel(file_path, dct):
    with pd.ExcelWriter(file_path) as writer:
        for k, v in dct.items():
            v.to_excel(writer, sheet_name=k)

def analyze(folder, path):
    ks = pd.read_pickle(os.path.join(folder, path))
    dct1 = descriptive(ks)
    dct2 = weight_vs_topx(ks)
    path1 = os.path.join(folder, 'KS_descriptive_stats.xlsx')
    path2 = os.path.join(folder, 'KS_weights_vs_topX.xlsx')
    export_to_excel(path1, dct1)
    export_to_excel(path2, dct2)

analyze(folder, path)
