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
    '_init_modules',
    'copy_samples_across_models',
    'copy_samples_wthin_country',
    'get_baseline',
    'get_module_models',
    'get_spearman',
    'get_uncertainties',
    'simulate_module_models',
    )


# %%

def _init_modules(module_name, include_data_path=False):
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

def copy_samples_across_models(models, exclude=()):
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
    Parameters
    ----------
    models : iterable(obj)
        List of models where samples will be copied across.
    exclude : list(str)
        Name of the parameters to be excluded from sample copying
        (e.g., country-specific parameters that should be different for each model).
    '''
    for i, model in enumerate(models[1:]):
        copied = models[:i+1]
        for j, original in enumerate(copied):
            copied_models = copied[:j]
            exclude_params = sum([list(m.parameters) for m in copied_models], [])
            exclude_params.extend([p for p in original.parameters if p.name in exclude])
            copy_samples(original, model,
                         exclude=exclude_params,
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

def get_module_models(
        module,
        create_general_model_func,
        create_country_specific_model_func,
        system_IDs=(),
        countries=(),
        country_specific_inputs=None,
        include_general_model=True,
        load_cached_data=False,
        general_model_kwargs={},
        country_specific_model_kwargs={},
        ):
    scores_path = os.path.join(path, f'{module}/scores')
    model_data_path = os.path.join(scores_path, 'model_data.pckl')
    model_dct = {}
    for sys_ID in system_IDs:
        # Country-specific ones
        model = None
        for n, country in enumerate(countries):
            try: country_data = country_specific_inputs.get(country)
            except: country_data = None
            kwargs = {
                'ID': sys_ID,
                'country': country,
                'country_data': country_data,
                'model': model
                }
            kwargs.update(country_specific_model_kwargs)
            # Reuse the model, just update parameters
            model = create_country_specific_model_func(**kwargs)
            model_key = get_model_key(model)
            model_dct[f'{model_key}_{country}'] = model
        # A non-country-specific general model
        if include_general_model:
            model = create_general_model_func(
                sys_ID,
                flowsheet=model.system.flowsheet,
                **general_model_kwargs,
                )
            model_dct[f'{model_key}_general'] = model
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
def get_spearman(model_dct, p_sig=0.05, spearman_path_prefix=''):
    spearman_rho_dct = {}
    spearman_rho_sig_dct = {}
    spearman_p_dct = {}
    for key, model in model_dct.items():
        flowsheet_ID, country = key.split('_')
        rho, p = model.spearman_r()
        spearman_rho_dct[key] = rho
        rho_sig = rho.where(p<=p_sig, other='')
        spearman_rho_sig_dct[key] = rho_sig
        spearman_p_dct[key] = p

    if spearman_path_prefix:
        writer_rho = pd.ExcelWriter(f'{spearman_path_prefix}_rho.xlsx')
        writer_rho_sig = pd.ExcelWriter(f'{spearman_path_prefix}_rho_sig.xlsx')
        writer_p = pd.ExcelWriter(f'{spearman_path_prefix}_p.xlsx')
        for name, df in spearman_rho_dct.items():
            df.to_excel(writer_rho, sheet_name=name)
            spearman_rho_sig_dct[name].to_excel(writer_rho_sig, sheet_name=name)
            spearman_p_dct[name].to_excel(writer_p, sheet_name=name)
        writer_rho.save()
        writer_rho_sig.save()
        writer_p.save()

    spearman_dct = {
        'rho': spearman_rho_dct,
        'rho_sig': spearman_rho_sig_dct,
        'p': spearman_p_dct,
        }
    return spearman_dct


# %%

@time_printer
def get_uncertainties(
        model_dct, N, seed=None,
        sample_hook_func=None, # function for sample processing
        pickle_path='',
        uncertainty_path='',
        return_model_dct=False,
        **kwargs):
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

def simulate_module_models(
        scores_path, model_dct, N,
        seed=None, sample_hook_func=None,
        return_model_dct=True,
        include_baseline=True,
        include_spearman=True,
        baseline_path='default',
        pickle_path='default',
        uncertainty_path='default',
        spearman_path_prefix='default',
        ):
    outs = []
    if include_baseline:
        if baseline_path=='default':
            baseline_path = os.path.join(scores_path, 'simulated_baseline.csv')
        baseline_df = get_baseline(model_dct=model_dct, file_path=baseline_path)
        outs.append(baseline_df)

    if pickle_path == 'default':
        pickle_path = os.path.join(scores_path, 'model_data.pckl')
    if uncertainty_path == 'default':
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
    outs.append(uncertainty_dct)

    if include_spearman:
        if spearman_path_prefix == 'default':
            spearman_path_prefix = os.path.join(scores_path, 'spearman')
        spearman_dct = get_spearman(model_dct, spearman_path_prefix=spearman_path_prefix)
        outs.append(spearman_dct)

    if return_model_dct: outs.append(model_dct)
    return outs