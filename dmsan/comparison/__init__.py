#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making for sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>
'''

import os
from dmsan.utils import (
    get_module_models,
    import_module_results,
    import_country_specifc_inputs,
    init_modules,
    simulate_module_models,
    )
from exposan.biogenic_refinery import create_country_specific_model as create_br_model
from exposan.new_generator import create_country_specific_model as create_ng_model
from exposan.reclaimer import create_country_specific_model as create_re_model

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

file_path = os.path.join(data_path, 'country_specific_inputs.csv')
country_specific_inputs  = import_country_specifc_inputs(file_path=file_path)
countries = country_specific_inputs.keys()

def get_models(
        module=module,
        countries=countries,
        country_specific_inputs=country_specific_inputs,
        load_cached_data=False,
        ):
    model_dct = get_module_models(
        module=module,
        create_country_specific_model_func=create_br_model,
        system_IDs=('A', 'B'),
        countries=countries,
        country_specific_inputs=country_specific_inputs,
        load_cached_data=load_cached_data,
        )
    model_dct.update(get_module_models(
        module=module,
        create_country_specific_model_func=create_ng_model,
        system_IDs=('A', 'B'),
        countries=countries,
        country_specific_inputs=country_specific_inputs,
        load_cached_data=load_cached_data,
        ))
    model_dct.update(get_module_models(
        module=module,
        create_country_specific_model_func=create_re_model,
        system_IDs=('B', 'C'),
        countries=countries,
        country_specific_inputs=country_specific_inputs,
        load_cached_data=load_cached_data,
        ))
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
        countries=countries, *,
        N,
        seed=None,
        module=module,
        country_specific_inputs=country_specific_inputs,
        ):
    model_dct = get_models(
        module=module,
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