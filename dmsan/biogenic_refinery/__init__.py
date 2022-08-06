#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making for sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>

    Hannah Lohman <hlohman94@gmail.com>
'''

import os
from exposan.biogenic_refinery import create_country_specific_model
from dmsan.utils import (
    get_module_models,
    import_module_results,
    init_modules,
    simulate_module_models,
    )

__all__ = (
    'scores_path',
    'results_path',
    'figures_path',
    'get_models',
    'import_results',
    'simulate_models',
    )

module = os.path.split(os.path.dirname(__file__))[-1]
system_IDs = ('A', 'B', 'C', 'D')
scores_path, results_path, figures_path = init_modules(module)


def get_models(
        module=module,
        system_IDs=system_IDs,
        countries=(),
        country_specific_inputs=None,
        load_cached_data=False,
        ):
    model_dct = get_module_models(
        module=module,
        create_country_specific_model_func=create_country_specific_model,
        system_IDs=system_IDs,
        countries=countries,
        country_specific_inputs=country_specific_inputs,
        load_cached_data=load_cached_data,)
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


def simulate_models(
        countries,
        N,
        seed=None,
        module=module,
        system_IDs=system_IDs,
        country_specific_inputs=None
        ):
    model_dct = get_models(
        module=module,
        system_IDs=system_IDs,
        countries=countries,
        country_specific_inputs=country_specific_inputs,
        load_cached_data=False,
        )

    baseline_df, uncertainty_dct = simulate_module_models(
        scores_path=scores_path,
        model_dct=model_dct,
        N=N,
        seed=seed)
    return baseline_df, uncertainty_dct