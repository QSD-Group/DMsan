#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making of sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/DMsan/blob/main/LICENSE.txt
for license details.
'''

import os, pandas as pd, qsdsan as qs
from qsdsan.utils import (
    copy_samples,
    load_data,
    load_pickle,
    save_pickle,
    time_printer,
    )

from . import path

__all__ = (
    'copy_samples_across_models',
    'copy_samples_wthin_country',
    'get_baseline',
    'get_module_models',
    'get_spearman',
    'get_uncertainties',
    'init_modules',
    'simulate_module_models',
    )


# %%

def copy_samples_across_models(models):
    '''
    Ensure consistent values for the same parameter across models.

    The idea is basically:

        .. code::

            copy_samples(modelA, modelB)

            copy_samples(modelA, modelC)
            copy_samples(modelB, modelC, exclude=modelA.parameters)

            copy_samples(modelA, modelD)
            copy_samples(modelB, modelD, exclude=modelA.parameters)
            copy_samples(modelC, modelD, exclude=(*modelA.parameters, *modelB.parameters)

            ...
    '''
    for i, model in enumerate(models[1:]):
        copied = models[:i+1]
        for j, original in enumerate(copied):
            exclude = copied[:j]
            copy_samples(original, model,
                         exclude=sum([list(m.parameters) for m in exclude], []),
                         only_same_baseline=True)

            # # To see what's being copied and what's being excluded
            # print('new: ', model.system.flowsheet.ID)
            # print('original: ', original.system.flowsheet.ID)
            # print('exclude: ', [m.system.flowsheet.ID for m in exclude], '\n\n')


def copy_samples_wthin_country(model_dct):
    '''Ensure consistent values for the same parameter across models for the same country.'''
    country_model_dct = {}
    for key, model in model_dct.items():
        flowsheet_ID, country = key.split('_')
        if not country in country_model_dct: country_model_dct[country] = {flowsheet_ID: model}
        else: country_model_dct[country][flowsheet_ID] = model
    for models in country_model_dct.values():
        copy_samples_across_models(list(models.values()))


# %%

get_model_key = lambda model: model.system.flowsheet.ID # brA, etc.
get_model_dct = lambda models: {get_model_key(model): model for model in models}

def update_settings(model):
    cmps = qs.get_components()
    if cmps is not model.system.units[0].components:
        qs.set_thermo(model.system.units[0].components)

    if qs.main_flowsheet is not model.system.flowsheet:
        qs.main_flowsheet.set_flowsheet(model.system.flowsheet)


def get_baseline(model_dct, file_path=''):
    df = pd.DataFrame()
    for key, model in model_dct.items():
        update_settings(model)
        df[key] = model.metrics_at_baseline()

    if file_path:
        sep = '\t' if file_path.endswith('.tsv') else ','
        df.to_csv(file_path, sep=sep)
    return df


# %%

def get_module_models(module,
                      create_general_model_func,
                      create_country_specific_model_func,
                      system_IDs=(),
                      countries=(),
                      country_specific_inputs=None,
                      load_cached_data=False,):
    scores_path = os.path.join(path, f'{module}/scores')
    model_data_path = os.path.join(scores_path, 'model_data.pckl')
    model_dct = {}
    for sys_ID in system_IDs:
        # Non-country-specific model
        model = create_general_model_func(sys_ID)
        model_key = get_model_key(model)
        model_dct[f'{model_key}_general'] = model
        for n, country in enumerate(countries):
            try: country_data = country_specific_inputs.get(country)
            except: country_data = None
            kwargs = {
                'ID': sys_ID,
                'country': country,
                'country_data': country_data,
                }
            # Reuse the model, just update parameters
            kwargs['model'] = model
            model = create_country_specific_model_func(**kwargs)
            model_dct[f'{model_key}_{country}'] = model

    if load_cached_data:
        if not os.path.isfile(model_data_path):
            raise FileNotFoundError(f'No existing model data found for module "{module}" '
                                    f'and country "{country}", '
                                    'please run the model and save the data first.')
        data = load_pickle(model_data_path)
        for key, model in model_dct.items():
            model._samples, model.table = data[key].values()
    return model_dct


# %%

# Not in the pickle file as it's easy to reproduce once have the model
@time_printer
def get_spearman(model_dct, spearman_path=''):
    spearman_dct = {}
    for key, model in model_dct.items():
        flowsheet_ID, country = key.split('_')
        rho, p = model.spearman_r()
        spearman_dct[key] = rho

    if spearman_path:
        writer = pd.ExcelWriter(spearman_path)
        for name, df in spearman_dct.items():
            df.to_excel(writer, sheet_name=name)
        writer.save()

    return spearman_path


# %%

@time_printer
def get_uncertainties(model_dct, N, seed=None,
                      sample_hook_func=None, # function for sample processing
                      pickle_path='',
                      uncertainty_path='',
                      return_model_dct=False,
                      ):
    for model in model_dct.values():
        samples = model.sample(N=N, seed=seed, rule='L')
        model.load_samples(samples)
    if sample_hook_func: sample_hook_func(model_dct)

    uncertainty_dct = {}
    for key, model in model_dct.items():
        flowsheet_ID, country = key.split('_')
        key = f'{get_model_key(model)}_{country}'

        df = model.table
        param_col = [col for col in df.columns[0: len(model.parameters)]]
        uncertainty_dct[f'{key}_params'] = model.table[param_col]

        print(f'uncertainties for model: {key}')
        update_settings(model)
        model.evaluate()

        uncertainty_dct[f'{key}_results'] = \
            model.table.iloc[:, len(model.parameters):]

    if pickle_path:
        # Cannot just save the `Model` object as a pickle file
        # because it contains local functions
        data = {}
        for key in uncertainty_dct.keys(): # brA_China_param, brA_China_results, etc.
            model_dct_key, kind = '_'.join(key.split('_')[:-1]), key.split('_')[-1] # brA_China, params
            if kind == 'params': continue # each model will appear twice, just need to save once
            model = model_dct[model_dct_key]
            data[model_dct_key] = {
                'samples': model._samples,
                'table': model.table
                }
        save_pickle(data, pickle_path)

    if uncertainty_path:
        writer = pd.ExcelWriter(uncertainty_path)
        for name, df in uncertainty_dct.items():
            df.to_excel(writer, sheet_name=name)
        writer.save()

    if return_model_dct: return uncertainty_dct, model_dct
    return uncertainty_dct


# %%

def import_country_specifc_inputs(file_path, return_as_dct=True):
    df = load_data(file_path)
    if not return_as_dct: return df

    data_dct = {}
    for country in df.columns:
        series = getattr(df, country)
        data_dct[country] = {k: series[k] for k in series.index}

    return data_dct


# %%

def init_modules(module_name, include_data_path=False):
    module_path = os.path.join(path, module_name)
    dirnames = ['scores', 'results', 'figures']
    if include_data_path: dirnames.insert(0, 'data')
    paths = []
    for dirname in dirnames:
        p = os.path.join(module_path, dirname)
        paths.append(p)
        if not os.path.isdir(p): os.mkdir(p)
    return paths


# %%

def simulate_module_models(scores_path, model_dct, N, seed=None, sample_hook_func=None):
    baseline_path = os.path.join(scores_path, 'simulated_baseline.csv')
    baseline_df = get_baseline(model_dct=model_dct, file_path=baseline_path)

    pickle_path = os.path.join(scores_path, 'model_data.pckl')
    uncertainty_path = os.path.join(scores_path, 'simulated_uncertainties.xlsx')
    uncertainty_dct, model_dct = get_uncertainties(
        model_dct=model_dct,
        N=N,
        seed=seed,
        sample_hook_func=copy_samples_wthin_country,
        pickle_path=pickle_path,
        uncertainty_path=uncertainty_path,
        return_model_dct=True,
        )

    spearman_path = os.path.join(scores_path, 'spearman.xlsx')
    spearman_dct = get_spearman(model_dct, spearman_path=spearman_path)
    return baseline_df, uncertainty_dct, spearman_dct