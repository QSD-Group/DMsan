#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making for sanitation and resource recovery systems

This module is developed by:
    Yalin Li <mailto.yalin.li@gmail.com>
    Hannah Lohman <hlohman94@gmail.com>

Run this module to save the results to the /results folder to avoid repeating
simulating the system.

TODOs: move functions that can be generalized into a `utils` module
'''

import os, numpy as np, pandas as pd
from qsdsan.utils import time_printer, load_pickle, save_pickle, copy_samples
from exposan.biogenic_refinery import create_country_specific_model as create_br_country_specific_model
from exposan.new_generator import create_country_specific_model as create_ng_country_specific_model
from exposan.reclaimer import create_country_specific_model as create_re_country_specific_model
from dmsan.comparison import scores_path

# Comment these out if want to see all warnings
import warnings
warnings.filterwarnings(action='ignore')

__all__ = ('rebuild_models', 'get_baseline', 'get_uncertainties')


# %%

# =============================================================================
# Util functions for reloading saved models
# =============================================================================

get_model_key = lambda model: f'{model.system.flowsheet.ID}{model.system.ID[-1]}' # brA, etc.
get_model_dct = lambda models: {get_model_key(model): model for model in models}

def get_model(create_model_func, country, N, seed=None, rule='L'):
    if seed: np.random.seed(seed)

    # Biogenic refinery
    br_modelA = create_br_country_specific_model('A', country)
    br_modelB = create_br_country_specific_model('B', country)

    # NEWgenerator
    ng_modelA = create_ng_country_specific_model('A', country)
    ng_modelB = create_ng_country_specific_model('B', country)

    # Reclaimer
    re_modelB = create_re_country_specific_model('B', country)
    re_modelC = create_re_country_specific_model('C', country)

    models = [
        br_modelA, br_modelB,
        ng_modelA, ng_modelB,
        re_modelB, re_modelC,
        ]

    for model in models:
        samples = model.sample(N, rule)
        model.load_samples(samples)

    #!!! Need to consider parameters that are the same across systems
    copy_samples(br_modelA, br_modelB)
    copy_samples(ng_modelA, ng_modelB)
    copy_samples(re_modelB, re_modelC)

    return get_model_dct(models)


def rebuild_models(country):
    path = os.path.join(scores_path, f'{country}/model_data.pckl')
    data = load_pickle(path)

    models = get_model(*data['inputs'])
    for model in models:
        key = get_model_key(model)
        model._samples = data['samples'][key]
        model._samples = data['tables'][key]

    return get_model_dct(models)


# %%

# =============================================================================
# Baseline and uncertainties
# =============================================================================

def get_baseline(model_dct, file_path=''):
    df = pd.DataFrame()
    for key, model in model_dct.items():
        df[key] = model.metrics_at_baseline()

    if file_path:
        sep = '\t' if file_path.endswith('.tsv') else ''
        df.to_csv(file_path, sep=sep)
    return df


# %%

# =============================================================================
# Add uncertainties
# =============================================================================

@time_printer
def get_uncertainties(country, N, seed=None, rule='L',
                      pickle_path='', param_path='', result_path=''):
    model_dct = get_model(country, N, seed, rule)
    uncertainty_dct = {}

    for key, model in model_dct.items():
        df = model.table
        param_col = [col for col in df.columns[0: len(model.parameters)]]
        uncertainty_dct[f'{key}-param'] = model.table[param_col]

        model.evaluate()
        uncertainty_dct[f'{key}-results'] = \
            model.table.iloc[:, len(model.parameters):]

    if param_path:
        dfs = dict.fromkeys(model_dct.keys())
        for key, model in model_dct.items():
            df = pd.DataFrame()
            parameters = [i for i in model.table.columns[:len(model.parameters)]]
            parameters.sort(key=lambda i: i[0][-2:])
            df['Parameters'] = parameters
            df['DV'] = df['T'] = df['RR'] = df['Env'] = df['Econ'] = df['S'] = ''
            dfs[key] = df

        writer = pd.ExcelWriter(param_path)
        for sys_ID, df in dfs.items():
            df.to_excel(writer, sheet_name=sys_ID)
        writer.save()

    if pickle_path:
        # Cannot just save the `Model` object as a pickle file
        # because it contains local functions
        data = {
            'inputs': [N, country, seed, rule],
            'samples': {get_model_key(model): model.samples for model in model_dct.values()},
            'tables': {get_model_key(model): model.table for model in model_dct.values()},
            }
        save_pickle(data, pickle_path)

    if result_path:
        writer = pd.ExcelWriter(result_path)
        for sys_ID, df in uncertainty_dct.items():
            df.to_excel(writer, sheet_name=sys_ID)
        writer.save()

    return uncertainty_dct


# %%

# =============================================================================
# Run all simulations
# =============================================================================

def run_simulations(country, N, seed=None):
    global baseline_df, uncertainty_dct
    country_folder = os.path.join(scores_path, country)
    # Create the folder if there isn't one already
    if not os.path.isdir(country_folder): os.mkdir(country_folder)
    baseline_path = os.path.join(country_folder, 'sys_baseline.tsv')
    param_path = os.path.join(country_folder, 'parameters.xlsx')
    pickle_path = os.path.join(country_folder, 'model_data.pckl')
    uncertainty_path = os.path.join(country_folder, 'sys_uncertainties.xlsx')

    baseline_df = get_baseline(file_path=baseline_path)
    uncertainty_dct = get_uncertainties(country=country, N=N, seed=seed,
                                        param_path=param_path,
                                        pickle_path=pickle_path,
                                        result_path=uncertainty_path)


if __name__ == '__main__':
    for country in ('China', 'India', 'Senegal', 'South Africa', 'Uganda'):
        run_simulations(country, N=20, seed=3221)