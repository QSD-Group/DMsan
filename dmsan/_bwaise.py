#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 13:31:44 2021

@author: Yalin Li <zoe.yalin.li@gmail.com>

"""

'''
TODO: figure out why ReCiPe gives so much credit to fertilizers
'''

# Run this module to save the results to the /data folder to avoid repeating
# simulating the system

import os
import pandas as pd
from exposan import bwaise as bw

__all__ = ('get_baseline', 'save_baseline', 'get_uncertainties', 'save_uncertainties')

data_path = os.path.join(os.path.dirname(__file__), 'data')


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


def save_baseline(path=''):
    if not path:
        path = os.path.join(data_path, 'lca_baseline.tsv')
    sep = '\t' if path.endswith('.tsv') else ''
    baseline_df = get_baseline()
    baseline_df.to_csv(os.path.join(data_path, 'bwaise_baseline.tsv'), sep=sep)
    return baseline_df

baseline_df = save_baseline()


# %%

# =============================================================================
# Add uncertainties
# =============================================================================

def get_uncertainties():
    from exposan.bwaise.models import update_metrics, update_LCA_CF_parameters, run_uncertainty
    models = modelA, modelB, modelC = bw.modelA, bw.modelB, bw.modelC

    for model in models:
        model = update_LCA_CF_parameters(model, 'new')
        model = update_metrics(model, 'new')

    uncertainty_dct = {}
    for model in models:
        uncertainty_dct[model.system.ID] = run_uncertainty(model, N=10)['data']
    return uncertainty_dct

def save_uncertainties(path=''):
    if not path:
        path = os.path.join(data_path, 'bwaise_uncertainties.xlsx')
    uncertainty_dct = get_uncertainties()
    writer = pd.ExcelWriter(path)

    for sys_ID, df in uncertainty_dct.items():
        columns = [col for col in df.columns
                   if 'recovery' in col[0].lower() and 'total' in col[1].lower()]
        columns += [col for col in df.columns if 'net cost' in col[1].lower()]
        columns += [col for col in df.columns if 'net emission' in col[1].lower()]
        new_df = df[columns]
        new_df.to_excel(writer, sheet_name=sys_ID)
    writer.save()

    return uncertainty_dct

uncertainty_dct = save_uncertainties()