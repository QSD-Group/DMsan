#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yalin Li <mailto.yalin.li@gmail.com>

Run this module to save the results to the /results folder to avoid repeating
simulating the system.
"""


import os
import numpy as np
import pandas as pd
from qsdsan.utils import time_printer, load_pickle, save_pickle, copy_samples
from exposan import bwaise as bw
from dmsan.bwaise import scores_path

# Comment these out if want to see all warnings
import warnings
warnings.filterwarnings(action='ignore')

__all__ = ('rebuild_models', 'get_baseline', 'get_uncertainties')


# %%

# =============================================================================
# Util functions for reloading saved models
# =============================================================================

def get_model(N, seed=None, rule='L', lca_perspective='H'):
    models = modelA, modelB, modelC = [bw.create_model(ID, lca_kind='new') for ID in ('A', 'B', 'C')]

    if seed: np.random.seed(seed)
    pers = ['I', 'H', 'E']
    pers.remove(lca_perspective)

    for model in models:
        # Only do ReCiPe, hierarchist (H) perspective
        model.set_parameters([i for i in model.parameters
                              if not (f' {pers[0]} ' in i.name or
                                      f' {pers[1]} ' in i.name or
                                      'global warming' in i.name)])
        get_metric = lambda name: [i for i in model.metrics if i.name==name]
        mRR = sum([get_metric(f'Total {i}') for i in ('N', 'P', 'K')], [])
        # mRR = sum([get_metric(f'Total {i}') for i in ('N', 'P', 'K', 'COD')], [])
        mRR += get_metric('Gas COD') # energy, only gas
        mEnv = [i for i in model.metrics
                if ('Net emission' in i.name and f'{lca_perspective}_' in i.name)]
        mEnv.sort(key=lambda i: i.name_with_units)
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
# Simulation functions
# =============================================================================

def get_baseline(models, path=''):
    baseline_dct = {}
    for model in models:
        baseline_dct[model.system.ID] = model.metrics_at_baseline()
    df = pd.DataFrame.from_dict(baseline_dct)
    if path: df.to_csv(path)
    return df


@time_printer
def get_uncertainties(models, N=1000, seed=None, rule='L', lca_perspective='H',
                      pickle_path='', param_path='', result_path=''):
    uncertainty_dct = {}

    for model in models:
        df = model.table
        param_col = [col for col in df.columns[0: len(model.parameters)]]
        uncertainty_dct[f'{model.system.ID}-param'] = model.table[param_col]

        model.evaluate()
        uncertainty_dct[f'{model.system.ID}-results'] = \
            model.table.iloc[:, len(model.parameters):]

    if param_path:
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

# Wrapper function
def run_simulations(models=None, baseline=False, uncertainty=False, save=False):
    if save:
        baseline_path = os.path.join(scores_path, 'sys_baseline.csv')
        param_path = os.path.join(scores_path, 'parameters.xlsx')
        pickle_path = os.path.join(scores_path, 'model_data.pckl')
        uncertainty_path = os.path.join(scores_path, 'sys_uncertainties.xlsx')
    else:
        baseline_path = param_path = pickle_path = uncertainty_path = ''

    global baseline_df, uncertainty_dct
    if baseline: baseline_df = get_baseline(models, path=baseline_path)
    if uncertainty:
        N = model_kwargs.get('N', 1000)
        seed = model_kwargs.get('seed', 3221)
        uncertainty_dct = get_uncertainties(models, N=N, seed=seed,
                                            param_path=param_path,
                                            pickle_path=pickle_path,
                                            result_path=uncertainty_path)

if __name__ == '__main__':
    # models = rebuild_models()
    model_kwargs = dict(N=100, seed=3221, rule='L', lca_perspective='H')
    models = get_model(**model_kwargs)
    run_simulations(
        models=models,
        baseline=True,
        uncertainty=True,
        save=True,
        )
    # run_simulations(baseline=True, uncertainty=True)