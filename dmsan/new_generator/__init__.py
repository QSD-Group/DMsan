#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making for sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>

    Hannah Lohman <hlohman94@gmail.com>
'''

import os
from qsdsan.utils import copy_samples
from exposan.new_generator import create_country_specific_model
from dmsan.utils import (
    get_module_models,
    import_module_results,
    init_modules,
    run_module_model_simulations,
    )

__all__ = (
    'scores_path',
    'results_path',
    'figures_path',
    'get_models',
    'import_results',
    )

module = os.path.split(os.path.dirname(__file__))[-1]
scores_path, results_path, figures_path = init_modules(module)


def get_models(country, load_cached_data=False):
    model_dct = get_module_models(
        module=module,
        create_model_func=create_country_specific_model,
        country=country,
        load_cached_data=load_cached_data,
        sys_IDs=('A', 'B'))
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


def run_model_simulations(country, N, seed=None):
    model_dct = get_models(country=country, load_cached_data=False)
    ngA, ngB = model_dct.values()

    for model in (ngA, ngB):
        samples = model.sample(N, seed=seed, rule='L')
        model.load_samples(samples)
    copy_samples(ngA, ngB)

    country_folder = os.path.join(scores_path, country)
    baseline_df, uncertainty_dct = run_module_model_simulations(country_folder, model_dct)
    return baseline_df, uncertainty_dct