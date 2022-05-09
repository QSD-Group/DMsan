#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Exposition of sanitation and resource recovery systems

This module is developed by:
    Hannah Lohman <hlohman94@gmail.com>
    Yalin Li <zoe.yalin.li@gmail.com>

Run this module to save the results to the /results folder to avoid repeating
simulating the system.
'''


import os
import numpy as np
import pandas as pd
from qsdsan.utils import time_printer, load_pickle, save_pickle, copy_samples
from exposan import bwaise as bw
from exposan import reclaimer as re
from dmsan.gates import scores_path

# Comment these out if want to see all warnings
import warnings
warnings.filterwarnings(action='ignore')

__all__ = ('rebuild_models', 'get_baseline', 'get_uncertainties')


# %%

# =============================================================================
# Util functions for reloading saved models
# =============================================================================

from exposan.reclaimer.country_specific import create_country_specific_model

def get_model(N, seed=None, rule='L'):
    models = []
    for country in ('China', 'XXX', 'XXX'):
        for sys_ID in ('sysA', 'sysB',):
            models.append(create_country_specific_model((sys_ID, country)))
    if seed:
        np.random.seed(seed)
    for model in models:
        get_metric = lambda name: [i for i in model.metrics if i.name==name]
        mRR = sum([get_metric(f'Total {i}') for i in ('N', 'P', 'K')], [])
        # mRR = sum([get_metric(f'Total {i}') for i in ('N', 'P', 'K', 'COD')], [])
        mRR += get_metric('Gas COD') # energy, only gas
        mEnv = [get_metric(name) for name in ('H_Resources', 'H_Health', 'H_Ecosystems')]
        # mEnv.sort(key=lambda i: i.name_with_units)  # I do not need this code
        mEcon = get_metric('Annual net cost')
        model.metrics = mRR + mEnv + mEcon

        samples = model.sample(N, rule)
        model.load_samples(samples)

    copy_samples(modelA, modelB)
    copy_samples(modelA, modelC)
    copy_samples(modelB, modelC, exclude=modelA.parameters)

    return modelA, modelB, modelC


def rebuild_models(path=''):
    path = path if path else os.path.join(scores_path, 'model_data.pckl')
    data = load_pickle(path)

    modelA, modelB, modelC = get_model(*data['inputs'])
    modelA._samples, modelB._samples, modelC._samples = data['samples']
    modelA.table, modelB.table, modelC.table = data['tables']

    return modelA, modelB, modelC


# %%

# =============================================================================
# Baseline with Hierarchist LCA
# =============================================================================

def get_cap_yr_pts(lca):
    impact_dct = lca.get_total_impacts()
    ratio = lca.lifetime * bw.systems.get_ppl(lca.system.ID[-1])
    for k, v in impact_dct.items():
        impact_dct[k] = v / ratio
    return impact_dct


def get_baseline(file_path=''):
    baseline_dct = {'sysA': [], 'sysB': [], 'sysC': []}
    inds = ['H_Ecosystems', 'H_Health', 'H_Resources']    # make sure this is in the same order always
    idxs = pd.MultiIndex.from_tuples([
        *zip(('Net recovery',)*4, ('N', 'P', 'K', 'energy')),
        *zip(('LCA results',)*len(inds), inds),
        ('TEA results', 'Net cost'),
        ])
    df = pd.DataFrame(index=idxs)

    sys_dct = bw.systems.sys_dct
    for sys in (bw.sysA, bw.sysB, bw.sysC):
        data = baseline_dct[sys.ID]
        func_dct = bw.systems.get_summarizing_functions(sys)
        for i in ('N', 'P', 'K'):
            data.append(func_dct[f'get_tot_{i}_recovery'](sys, i))
        data.append(func_dct['get_gas_COD_recovery'](sys, 'COD')) # energy, only gas

        lca = sys_dct['LCA'][sys.ID]
        data.extend((i for i in get_cap_yr_pts(lca).values()))

        tea = sys_dct['TEA'][sys.ID]
        ppl = sys_dct['ppl'][sys.ID]
        data.append(func_dct['get_annual_net_cost'](tea, ppl))
        df[sys.ID] = data

    if file_path:
        sep = '\t' if file_path.endswith('.tsv') else ''

        df.to_csv(file_path, sep=sep)
    return df

baseline_path = os.path.join(scores_path, 'sys_baseline.tsv')


# %%

# =============================================================================
# Add uncertainties
# =============================================================================

@time_printer
def get_uncertainties(N, seed=None, rule='L', lca_perspective='H',
                      pickle_path='', param_path='', result_path=''):
    models = get_model(N, seed, rule)
    uncertainty_dct = {}

    for model in models:
        df = model.table
        param_col = [col for col in df.columns[0: len(model.parameters)]]
        uncertainty_dct[f'{model.system.ID}-param'] = model.table[param_col]

        model.evaluate()
        uncertainty_dct[f'{model.system.ID}-results'] = \
            model.table.iloc[:, len(model.parameters):]

    if param_path:
        # MAKE SURE I CALL MY SYSTEMS WHAT I WANT HERE AND IN SPREADSHEETS
        dfs = dict.fromkeys(('Alternative A', 'Alternative B', 'Alternative C'))
        for model in models:
            df = dfs[f'Alternative {model.system.ID[-1]}'] = pd.DataFrame()
            parameters = [i for i in model.table.columns[:len(model.parameters)]]
            parameters.sort(key=lambda i: i[0][-2:])
            df['Parameters'] = parameters
            df['DV'] = df['T'] = df['RR'] = df['Env'] = df['Econ'] = df['S'] = ''

        writer = pd.ExcelWriter(param_path)
        for sys_ID, df in dfs.items():
            df.to_excel(writer, sheet_name=sys_ID)
        writer.save()

    if pickle_path:
        # Cannot just save the `Model` object as a pickle file
        # because it contains local functions
        data = {
            'inputs': [N, seed, rule, lca_perspective],
            'samples': [i._samples for i in models],
            'tables': [i.table for i in models]
            }
        save_pickle(data, pickle_path)

    if result_path:
        writer = pd.ExcelWriter(result_path)
        for sys_ID, df in uncertainty_dct.items():
            df.to_excel(writer, sheet_name=sys_ID)
        writer.save()

    return uncertainty_dct


param_path = os.path.join(scores_path, 'parameters.xlsx')
pickle_path = os.path.join(scores_path, 'model_data.pckl')
uncertainty_path = os.path.join(scores_path, 'sys_uncertainties.xlsx')


# %%

# =============================================================================
# Run all simulations
# =============================================================================

def run_simulations():
    global baseline_df, uncertainty_dct
    baseline_df = get_baseline(file_path=baseline_path)
    uncertainty_dct = get_uncertainties(N=1000, seed=3221,
                                        param_path=param_path,
                                        pickle_path=pickle_path,
                                        result_path=uncertainty_path)

if __name__ == '__main__':
    run_simulations()