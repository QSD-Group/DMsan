#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making for sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>
'''

import os
from qsdsan.utils import copy_samples
from dmsan.utils import (
    get_module_models,
    import_module_results,
    init_modules,
    simulate_module_models,
    )
from exposan.biogenic_refinery import create_country_specific_model as create_br_model
from exposan.new_generator import create_country_specific_model as create_ng_model
from exposan.reclaimer import create_country_specific_model as create_re_model



# from dmsan.biogenic_refinery import get_models as get_br_models
# from dmsan.new_generator import get_models as get_ng_models
# from dmsan.reclaimer import get_models as get_re_models

__all__ = (
    'scores_path',
    'results_path',
    'figures_path',
    'get_models',
    'import_results',
    'simulate_models',
    )

module = os.path.split(os.path.dirname(__file__))[-1]
data_path, scores_path, results_path, figures_path = init_modules(module, include_data_path=True)


def get_models(country, module=module, load_cached_data=False):
    model_dct = get_module_models(
        module=module,
        create_model_func=create_br_model,
        country=country,
        load_cached_data=load_cached_data,
        sys_IDs=('A', 'B'))
    model_dct.update(get_module_models(
        module=module,
        create_model_func=create_ng_model,
        country=country,
        load_cached_data=load_cached_data,
        sys_IDs=('A', 'B')))
    model_dct.update(get_module_models(
        module=module,
        create_model_func=create_re_model,
        country=country,
        load_cached_data=load_cached_data,
        sys_IDs=('B', 'C')))
    return model_dct


def import_results(
        country,
        parameters=False,
        indicator_scores=False,
        ahp=False,
        mcda=False,
        uncertainty=False,
        sensitivity=None
        ):
    path = os.path.join(results_path, country)
    return import_module_results(
            path,
            parameters=parameters,
            indicator_scores=indicator_scores,
            ahp=ahp,
            mcda=mcda,
            uncertainty=uncertainty,
            sensitivity=sensitivity,
            )


def simulate_models(country, N, seed=None, module=module):
    model_dct = get_models(country=country, module=module, load_cached_data=False)
    models = list(model_dct.values())
    for model in models:
        samples = model.sample(N, seed=seed, rule='L')
        model.load_samples(samples)

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

            # # The idea is basically
            # brA = model_dct['brA']
            # brB = model_dct['brB']
            # ngA = model_dct['ngA']
            # ngB = model_dct['ngB']
            # reB = model_dct['reB']
            # reC = model_dct['reC']

            # copy_samples(brA, brB)

            # copy_samples(brA, ngA)
            # copy_samples(brB, ngA, exclude=brA.parameters)

            # copy_samples(brA, ngB)
            # copy_samples(brB, ngB, exclude=brA.parameters)
            # copy_samples(ngA, ngB, exclude=(*brA.parameters, *brB.parameters))

    country_folder = os.path.join(scores_path, country)
    baseline_df, uncertainty_dct = simulate_module_models(country_folder, model_dct)
    return baseline_df, uncertainty_dct