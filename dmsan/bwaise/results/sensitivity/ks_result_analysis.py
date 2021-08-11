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
    out_1 = {}
    out_2 = {}
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

def analyze(folder, path):
    ks = pd.read_pickle(os.path.join(folder, path))
    dct1, dct2 = descriptive(ks)
    path1 = os.path.join(folder, 'KS_descriptive_stats.xlsx')
    path2 = os.path.join(folder, 'KS_weights_vs_topX.xlsx')
    export_to_excel(path1, dct1)
    export_to_excel(path2, dct2)

analyze(folder, path)
