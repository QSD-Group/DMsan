#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making for sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/DMsan/blob/main/LICENSE.txt
for license details.
'''

import os
from dmsan.utils import (
    get_module_models,
    import_country_specifc_inputs,
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
data_path, scores_path, results_path, figures_path = init_modules(module, include_data_path=True)

file_path = os.path.join(data_path, 'country_specific_inputs.csv')
country_specific_inputs  = import_country_specifc_inputs(file_path=file_path)
countries = country_specific_inputs.keys()

def get_models(
        module=module,
        countries=countries,
        country_specific_inputs=country_specific_inputs,
        include_resource_recovery=False,
        load_cached_data=False,
        ):
    from exposan import biogenic_refinery as br
    br.INCLUDE_RESOURCE_RECOVERY = include_resource_recovery
    from exposan.biogenic_refinery import (
        create_model as create_br_model,
        create_country_specific_model as create_br_country_model,
        )
    model_dct = get_module_models(
        module=module,
        create_general_model_func=create_br_model,
        create_country_specific_model_func=create_br_country_model,
        system_IDs=('A', 'B'),
        countries=countries,
        country_specific_inputs=country_specific_inputs,
        load_cached_data=load_cached_data,
        )

    try: # check if has access to the private repository
        from exposan import new_generator as ng
        ng.INCLUDE_RESOURCE_RECOVERY = include_resource_recovery
        from exposan.new_generator import (
            create_model as create_ng_model,
            create_country_specific_model as create_ng_country_model,
            )
        model_dct.update(get_module_models(
            module=module,
            create_general_model_func=create_ng_model,
            create_country_specific_model_func=create_ng_country_model,
            system_IDs=('A', 'B'),
            countries=countries,
            country_specific_inputs=country_specific_inputs,
            load_cached_data=load_cached_data,
            ))
    except ImportError:
        from warnings import warn
        warn('Simulation for the NEWgenerator system (under non-disclosure agreement) is skipped, '
             'please set path to use the EXPOsan-private repository if you have access.')

    from exposan import reclaimer as re
    re.INCLUDE_RESOURCE_RECOVERY = include_resource_recovery
    from exposan.reclaimer import (
        create_model as create_re_model,
        create_country_specific_model as create_re_country_model,
        )
    model_dct.update(get_module_models(
        module=module,
        create_general_model_func=create_re_model,
        create_country_specific_model_func=create_re_country_model,
        system_IDs=('B', 'C'),
        countries=countries,
        country_specific_inputs=country_specific_inputs,
        load_cached_data=load_cached_data,
        ))
    return model_dct


def simulate_models(
        countries=countries, *,
        N,
        seed=None,
        module=module,
        country_specific_inputs=country_specific_inputs,
        include_resource_recovery=False,
        ):
    model_dct = get_models(
        module=module,
        countries=countries,
        country_specific_inputs=country_specific_inputs,
        include_resource_recovery=include_resource_recovery,
        load_cached_data=False,
        )
    return simulate_module_models(
        scores_path=scores_path,
        model_dct=model_dct,
        N=N,
        seed=seed
        )