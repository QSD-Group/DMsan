#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yalin Li <mailto.yalin.li@gmail.com>

Run this module to simulate the systems and save the results to the /results folder.
"""


import os, re, numpy as np, pandas as pd
from qsdsan.utils import time_printer, load_pickle, save_pickle, copy_samples
from exposan import bwaise as bw
from dmsan.bwaise import scores_path

# Comment these out if want to see all warnings
import warnings
warnings.filterwarnings(action='ignore')

indicator_scores_path = os.path.join(scores_path, 'other_indicator_scores.xlsx')
alt_names = pd.read_excel(indicator_scores_path, sheet_name='user_interface').system

# `lca_perspective` can be "I", "H", or "E" for
# individualist, hierarchist, or egalitarian
lca_perspective = 'H'
excluded_pers = [i for i in ['I', 'H', 'E'] if i!=lca_perspective]


# %%

# =============================================================================
# Util functions for reloading saved models
# =============================================================================

def get_model(N, seed=None, rule='L', lca_perspective=lca_perspective):
    models = modelA, modelB, modelC = [bw.create_model(ID, lca_kind='new') for ID in ('A', 'B', 'C')]
    if seed: np.random.seed(seed)

    for model in models:
        ##### Parameters #####
        lca_params = [p for p in model.parameters if p.element=='LCA']
        params = [p for p in model.parameters if not p in lca_params]
        lca_params = [p for p in lca_params
                      if not (f' {excluded_pers[0]} ' in p.name or
                              f' {excluded_pers[1]} ' in p.name or
                              'global warming' in p.name)]
        lca_params.sort(key=lambda p: p.name)

        params = params + lca_params
        model.set_parameters(params)

        ##### Metrics #####
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
def get_uncertainties(models, N=1000, seed=None, rule='L',
                      lca_perspective=lca_perspective,
                      pickle_path='', result_path=''):
    uncertainty_dct = {}

    for model in models:
        df = model.table
        param_col = [col for col in df.columns[0: len(model.parameters)]]
        uncertainty_dct[f'{model.system.ID}-param'] = model.table[param_col]

        model.evaluate()
        uncertainty_dct[f'{model.system.ID}-results'] = \
            model.table.iloc[:, len(model.parameters):]

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

# Test whether an uncertainty parameter affects RR, Env, and Econ scores.
@time_printer
def test_parameters(models, param_path=''):
    def refresh_parameters(model): # some parameters will affect other parameters
        for p in model.parameters:
            p.setter(p.baseline)

    def format_dist(p):
        p_string = str(p.distribution)
        splitted = re.split(r'\(|\)|,|=', p_string)
        splitted = [i.lstrip() for i in splitted]
        dist = [splitted[0]]
        dist.extend([i for i in splitted if (i.isnumeric() or '.' in i)])
        if dist[0] == 'Uniform':
            return dist[0][0], float(dist[1]), '', float(dist[2])
        elif dist[0] == 'Triangle':
            return 'T', float(dist[1]), float(dist[2]), float(dist[3])
        elif dist[0] in ('Trunc', 'Normal'):
            return 'N', float(p.distribution.lower), \
                float(splitted[splitted.index('mu')+1]), float(p.distribution.upper)
        else:
            raise ValueError(f'Distribution is {p_string}, not recognized.')

    pnames = [[p.name_with_units for p in model.parameters] for model in models]

    param_dct = {}
    for m in models:
        df = pd.DataFrame(columns=('Parameters', 'DV', 'T', 'RR', 'Env', 'Econ', 'S'))
        parameters = [i for i in m.table.columns[:len(m.parameters)]]
        parameters.sort(key=lambda i: i[0][-2:])
        df['Parameters'] = parameters
        m_baseline = m.metrics_at_baseline()

        for n, p in enumerate(m.parameters):
            alts = [] # check if the alternative system has this parameter
            for names in pnames: alts.append(True if p.name_with_units in names else False)

            p_baseline = p.baseline

            # Min
            p.baseline = p.distribution.lower.item() if not p.hook \
                else p.hook(p.distribution.lower.item())
            refresh_parameters(m)
            m_min = m.metrics_at_baseline()

            # Max
            p.baseline = p.distribution.upper.item() if not p.hook \
                else p.hook(p.distribution.upper.item())
            refresh_parameters(m)
            m_max = m.metrics_at_baseline()

            p.baseline = p_baseline # reset parameter values
            diff = (m_max-m_min)/m_baseline
            diffs = [np.abs(diff[:3]).sum(), np.abs(diff[4:-1]).sum(), np.abs(diff[-1])]

            idx = df[df.Parameters==p.index].index
            df.loc[idx, ['RR', 'Env', 'Econ']] = \
                [True if abs(i)>=1e-6 else False for i in diffs] # make it False for np.nan

            df.loc[idx, ['Baseline']] = p_baseline
            df.loc[idx, ['Distribution', 'Lower', 'Midpoint', 'Upper']] = format_dist(p)
            df.loc[idx, alt_names] = alts

        sys_ID = m.system.ID
        param_dct[sys_ID] = df

    if param_path:
        writer = pd.ExcelWriter(param_path)
        for sys_ID, df in param_dct.items():
            df.to_excel(writer, sheet_name=f'Alternative {sys_ID[-1]}')
        writer.save()

    return param_dct


# Wrapper function
def run_simulations(models=None, model_kwargs={},
                    baseline=False, uncertainty=False, parameters=False, save=False):
    if save:
        baseline_path = os.path.join(scores_path, 'sys_baseline.csv')
        pickle_path = os.path.join(scores_path, 'model_data.pckl')
        uncertainty_path = os.path.join(scores_path, 'sys_uncertainties.xlsx')
        param_path = os.path.join(scores_path, 'parameters.xlsx')
    else:
        baseline_path = pickle_path = uncertainty_path = param_path = ''
    global baseline_df, uncertainty_dct
    if baseline: baseline_df = get_baseline(models, path=baseline_path)
    if uncertainty:
        N = model_kwargs.get('N', 1000)
        seed = model_kwargs.get('seed', 3221)
        uncertainty_dct = get_uncertainties(models, N=N, seed=seed,
                                            pickle_path=pickle_path,
                                            result_path=uncertainty_path)
    if parameters: test_parameters(models=models, param_path=param_path)


if __name__ == '__main__':
    # models = rebuild_models()
    model_kwargs = dict(N=100, seed=3221, rule='L')
    models = get_model(**model_kwargs)
    run_simulations(
        models=models,
        baseline=True,
        uncertainty=True,
        parameters=True,
        save=True,
        )
    # run_simulations(baseline=True, uncertainty=True)