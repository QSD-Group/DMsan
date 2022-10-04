#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making for sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>

    Hannah Lohman <hlohman94@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/DMsan/blob/main/LICENSE.txt
for license details.
'''

import os
from exposan import new_generator as ng
from exposan.new_generator import create_model, create_country_specific_model
from dmsan.utils import (
    _init_modules,
    get_module_models,
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
system_IDs = ('A', 'B')
scores_path, results_path, figures_path = _init_modules(module)


def get_models(
        module=module,
        system_IDs=system_IDs,
        countries=(),
        country_specific_inputs=None,
        include_resource_recovery=False,
        include_general_model=True,
        load_cached_data=False,
        ):
    ng.INCLUDE_RESOURCE_RECOVERY = include_resource_recovery
    model_dct = get_module_models(
        module=module,
        create_general_model_func=create_model,
        create_country_specific_model_func=create_country_specific_model,
        system_IDs=system_IDs,
        countries=countries,
        country_specific_inputs=country_specific_inputs,
        include_general_model=include_general_model,
        load_cached_data=load_cached_data,
        )
    return model_dct


def simulate_models(
        countries,
        N,
        seed=None,
        module=module,
        system_IDs=system_IDs,
        country_specific_inputs=None,
        include_resource_recovery=False,
        include_general_model=True,
        ):
    model_dct = get_models(
        module=module,
        system_IDs=system_IDs,
        countries=countries,
        country_specific_inputs=country_specific_inputs,
        include_resource_recovery=include_resource_recovery,
        include_general_model=include_general_model,
        load_cached_data=False,
        )
    return simulate_module_models(
        scores_path=scores_path,
        model_dct=model_dct,
        N=N,
        seed=seed
        )