#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making of sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>
'''

import os, pandas as pd
from qsdsan.utils import load_pickle, save_pickle, time_printer
from . import path

__all__ = (
    'get_baseline',
    'get_module_models',
    'get_uncertainties',
    'import_module_results',
    'init_modules',
    'run_module_model_simulations',
    )


# %%

get_model_key = lambda model: model.system.flowsheet.ID # brA, etc.
get_model_dct = lambda models: {get_model_key(model): model for model in models}

def get_baseline(models, file_path=''):
    df = pd.DataFrame()
    model_dct = models if isinstance(models, dict) else get_model_dct(models)
    for key, model in model_dct.items():
        df[key] = model.metrics_at_baseline()

    if file_path:
        sep = '\t' if file_path.endswith('.tsv') else ''
        df.to_csv(file_path, sep=sep)
    return df


# %%

def get_module_models(module, create_model_func, country, load_cached_data=False, sys_IDs=()):
    scores_path = os.path.join(path, f'{module}/scores')
    model_path = os.path.join(scores_path, f'{country}/model_data.pckl')
    models = [create_model_func(ID, country) for ID in sys_IDs]
    if load_cached_data:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'No existing model data found for module "{module}" '
                                    f'and country "{country}", '
                                    'please run the model and save the data first.')
        data = load_pickle(model_path)
        for model in models:
            key = get_model_key(model)
            model._samples = data['samples'][key]
            model._table = data['tables'][key]
    return get_model_dct(models)


# %%

@time_printer
def get_uncertainties(models, country, pickle_path='', param_path='', result_path=''):
    model_dct = models if isinstance(models, dict) else get_model_dct(models)

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
            'country': country,
            'samples': {get_model_key(model): model._samples for model in model_dct.values()},
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

def import_module_results(
        path, parameters=False, indicator_scores=False,
        ahp=False, mcda=False, uncertainty=False, sensitivity=None):
    loaded = dict.fromkeys(('param', 'tech_score', 'ahp', 'mcda',
                            'uncertainty', 'sensitivity'))

    if parameters:
        file_path = os.path.join(path, 'parameters.pckl')
        loaded['parameters'] = load_pickle(file_path)

    if indicator_scores:
        file_path = os.path.join(path, 'indicator_scores.pckl')
        loaded['indicator_scores'] = load_pickle(file_path)

    if ahp:
        file_path = os.path.join(path, 'ahp.pckl')
        loaded['ahp'] = load_pickle(file_path)

    if mcda:
        file_path = os.path.join(path, 'mcda.pckl')
        loaded['mcda'] = load_pickle(file_path)

    if uncertainty:
        file_path = os.path.join(path, 'uncertainty/performance_uncertainties.pckl')
        loaded['uncertainty'] = load_pickle(file_path)

    if sensitivity:
        file_path = os.path.join(path, f'sensitivity/performance_{sensitivity}_ranks.pckl')
        loaded['sensitivity'] = [load_pickle(file_path)]

        if sensitivity != 'KS':
            file_path = os.path.join(path, f'sensitivity/AHP_TOPSIS_{sensitivity}_scores.xlsx')
            loaded['sensitivity'].append(load_pickle(file_path))

    return loaded


# %%

def init_modules(module_name):
    module_path = os.path.join(path, module_name)
    scores_path = os.path.join(module_path, 'scores')
    results_path = os.path.join(module_path, 'results')
    figures_path = os.path.join(module_path, 'figures')
    for p in (scores_path, results_path, figures_path):
        if not os.path.isdir(p): os.mkdir(p)
    return scores_path, results_path, figures_path


# %%

def run_module_model_simulations(country_folder, model_dct):
    country = os.path.split(country_folder)[-1]
    # Create the folder if there isn't one already
    if not os.path.isdir(country_folder): os.mkdir(country_folder)
    baseline_path = os.path.join(country_folder, 'sys_baseline.tsv')
    param_path = os.path.join(country_folder, 'parameters.xlsx')
    pickle_path = os.path.join(country_folder, 'model_data.pckl')
    uncertainty_path = os.path.join(country_folder, 'sys_uncertainties.xlsx')
    baseline_df = get_baseline(models=model_dct, file_path=baseline_path)
    uncertainty_dct = get_uncertainties(models=model_dct,
                                        country=country,
                                        param_path=param_path,
                                        pickle_path=pickle_path,
                                        result_path=uncertainty_path)
    return baseline_df, uncertainty_dct