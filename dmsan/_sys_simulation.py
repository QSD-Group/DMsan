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
from qsdsan.utils.decorators import time_printer
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
        func_dct = bw.systems.get_summarizing_functions(sys)
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

baseline_df = save_baseline()


# %%

# =============================================================================
# Add uncertainties
# =============================================================================

def copy_samples(original, new, exclude=()):
    '''
    Copy samples of the shared parameters in the original model to the new model.
    Parameters in `exclude` will be excluded (i.e., not copied).
    '''
    col0 = original.table.columns.get_level_values(1)[:len(original.parameters)]
    col1 = new.table.columns.get_level_values(1)[:len(new.parameters)]
    shared = col0.intersection(col1)
    shared = shared.difference([i.name_with_units for i in exclude])
    idx0 = original.table.columns.get_locs([slice(None), shared])
    idx1 = new.table.columns.get_locs([slice(None), shared])
    new.table[new.table.columns[idx1]] = new._samples[:, idx1] \
        = original._samples[:, idx0]


@time_printer
def get_uncertainties(N=1000, seed=None, rule='L', file_path='', ):
    from exposan.bwaise.models import update_metrics, update_LCA_CF_parameters
    models = modelA, modelB, modelC = bw.modelA, bw.modelB, bw.modelC

    bw.update_lca_data('new')
    if seed:
        np.random.seed(seed)

    uncertainty_dct = {}
    for model in models:
        model = update_LCA_CF_parameters(model, 'new')

        # Only do ReCiPe, hierarchist (H) perspective
        model.set_parameters([i for i in model.parameters
                              if not (' I ' in i.name or
                                      ' E ' in i.name or
                                      'global warming' in i.name)])

        model = update_metrics(model, 'new')
        m1 = [i for i in model.metrics if (
            ('recovery' in i.element and 'Total' in i.name) or
            i.name=='Annual net cost')]
        m2 =[i for i in model.metrics if ('Net emission' in i.name and 'H_' in i.name)]
        m2.sort(key=lambda i: i.name_with_units)

        model.metrics = m1 + m2
        samples = model.sample(N, rule)
        model.load_samples(samples)

    copy_samples(modelA, modelB)
    copy_samples(modelA, modelC)
    copy_samples(modelB, modelC, exclude=modelA.parameters)

    for model in models:
        df = model.table
        param_col = [col for col in df.columns[0: len(model.parameters)]]
        uncertainty_dct[f'{model.system.ID}-param'] = model.table[param_col]

        # # Legacy code to select results
        # result_col = [col for col in df.columns
        #            if 'recovery' in col[0].lower() and 'total' in col[1].lower()]
        # result_col += [col for col in df.columns if 'net cost' in col[1].lower()]
        # result_col += [col for col in df.columns if 'net emission' in col[1].lower()]

        model.evaluate()
        uncertainty_dct[f'{model.system.ID}-results'] = \
            model.table.iloc[:, len(model.parameters):]

    if file_path:
        writer = pd.ExcelWriter(file_path)
        for sys_ID, df in uncertainty_dct.items():
            df.to_excel(writer, sheet_name=sys_ID)
        writer.save()

    return uncertainty_dct

file_path = os.path.join(data_path, 'bwaise_uncertainties.xlsx')
# file_path = '' # if don't want to save the file but want to see the results
uncertainty_dct = get_uncertainties(N=1000, seed=3221, file_path=file_path)