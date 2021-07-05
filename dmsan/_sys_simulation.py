#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 13:31:44 2021

@author: Yalin Li <zoe.yalin.li@gmail.com>

"""

# Run this module to save the results to the /data folder to avoid repeating
# simulating the system

import os
import numpy as np
import pandas as pd
from exposan import bwaise as bw
from dmsan import data_path

__all__ = ('get_baseline', 'save_baseline', 'get_uncertainties')


# %%

# =============================================================================
# Baseline for Individualist, Hierarchist, and Egalitarian (I/H/E)
# =============================================================================

def update_lca():
    bw.update_lca_data('new')
    lcas = lcaA, lcaB, lcaC = bw.lcaA, bw.lcaB, bw.lcaC

    indicators = []
    for cat in ('I', 'H', 'E'):
        temp = [i for i in lcaA.indicators if cat+'_' in i.ID and '_Total' in i.ID]
        temp.sort(key=lambda ind: ind.ID)
        indicators.extend(temp)

    for lca in lcas:
        lca.indicators = indicators

    return lcas


def get_cap_yr_pts(lca):
    impact_dct = lca.get_total_impacts()
    ratio = lca.lifetime * bw.systems.get_ppl(lca.system.ID[-1])
    for k, v in impact_dct.items():
        impact_dct[k] = v / ratio
    return impact_dct


def get_baseline():
    baseline_dct = {'sysA': [], 'sysB': [], 'sysC': []}
    lcas = update_lca()
    inds = lcas[0].indicators
    idxs = pd.MultiIndex.from_tuples([
        *zip(('Net recovery',)*4, ('N', 'P', 'K', 'energy')),
        ('TEA results', 'Net cost'),
        *zip(('LCA results',)*len(inds), inds)
        ])
    df = pd.DataFrame(index=idxs)

    sys_dct = bw.systems.sys_dct
    for sys in (bw.sysA, bw.sysB, bw.sysC):
        data = baseline_dct[sys.ID]
        func_dct = bw.systems.get_summarizing_fuctions(sys)
        for i in ('N', 'P', 'K'):
            data.append(func_dct[f'get_tot_{i}_recovery'](sys, i))
        data.append(func_dct['get_gas_COD_recovery'](sys, 'COD')) # energy

        tea = sys_dct['TEA'][sys.ID]
        ppl = sys_dct['ppl'][sys.ID]
        data.append(func_dct['get_annual_net_cost'](tea, ppl))

        lca = sys_dct['LCA'][sys.ID]
        data.extend((i for i in get_cap_yr_pts(lca).values()))
        df[sys.ID] = data
    return df


def save_baseline(file_path=''):
    if not file_path:
        file_path = os.path.join(data_path, 'lca_baseline.tsv')
    sep = '\t' if file_path.endswith('.tsv') else ''
    baseline_df = get_baseline()
    baseline_df.to_csv(os.path.join(data_path, 'bwaise_baseline.tsv'), sep=sep)
    return baseline_df

# baseline_df = save_baseline()


# %%

# =============================================================================
# Add uncertainties
# =============================================================================

def get_uncertainties(N=1000, seed=None, rule='L', file_path=''):
    from exposan.bwaise.models import update_metrics, update_LCA_CF_parameters
    models = modelA, modelB, modelC = bw.modelA, bw.modelB, bw.modelC

    if seed:
        np.random.seed(seed)

    uncertainty_dct = {}
    for model in models:
        model = update_LCA_CF_parameters(model, 'new')
        model = update_metrics(model, 'new')

        samples = model.sample(N, rule)
        model.load_samples(samples)
        df = model.table
        param_col = [col for col in df.columns[0: len(model.parameters)]]
        uncertainty_dct[f'{model.system.ID}-param'] = model.table[param_col]

        result_col = [col for col in df.columns
                   if 'recovery' in col[0].lower() and 'total' in col[1].lower()]
        result_col += [col for col in df.columns if 'net cost' in col[1].lower()]
        result_col += [col for col in df.columns if 'net emission' in col[1].lower()]

        model.evaluate()
        uncertainty_dct[f'{model.system.ID}-results'] = model.table[result_col]

    if file_path:
        writer = pd.ExcelWriter(file_path)
        for sys_ID, df in uncertainty_dct.items():
            df.to_excel(writer, sheet_name=sys_ID)
        writer.save()

    return uncertainty_dct

# TODO: figure out why GWP factors are in sysA's parameters
# TODO: discuss about the parameter selection
#       now 1.3-1.4K parameters, would probably still be hundreds with the gross total

file_path = os.path.join(data_path, 'bwaise_uncertainties.xlsx')
uncertainty_dct = get_uncertainties(N=10, seed=3221, file_path=file_path)