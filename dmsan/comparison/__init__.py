#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making for sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>
'''

import os
from exposan.biogenic_refinery import create_country_specific_model as create_br_country_specific_model
from exposan.new_generator import create_country_specific_model as create_ng_country_specific_model
from exposan.reclaimer import create_country_specific_model as create_re_country_specific_model
from dmsan.utils import get_module_models, import_module_results, init_modules

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
        create_model_func=create_br_country_specific_model,
        country=country,
        load_cached_data=load_cached_data,
        sys_IDs=('A', 'B'))
    model_dct = get_module_models(
        module=module,
        create_model_func=create_ng_country_specific_model,
        country=country,
        load_cached_data=load_cached_data,
        sys_IDs=('A', 'B'))
    model_dct = get_module_models(
        module=module,
        create_model_func=create_re_country_specific_model,
        country=country,
        load_cached_data=load_cached_data,
        sys_IDs=('B', 'C'))
    return model_dct


def import_results(country):
    return import_module_results(module, country)