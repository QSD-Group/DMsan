#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making of sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>

    Hannah Lohman <hlohman94@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/DMsan/blob/main/LICENSE.txt
for license details.

Run this module to save the results to the /scores folder to avoid repeating
simulating the system.
'''

from dmsan.reclaimer import simulate_models, get_models

# Comment these out if want to see all warnings
import warnings
warnings.filterwarnings(action='ignore')

system_IDs = ('B', 'C')
countries = ('China', 'India', 'Senegal', 'South Africa', 'Uganda')
N = 20
seed = 3221

if __name__ == '__main__':
    outs = simulate_models(countries=countries, N=N, seed=seed, include_resource_recovery=True)
    baseline_df, uncertainty_dct, spearman_dct, model_dct = outs

    # # To reload models
    # model_dct = get_models(
    #         system_IDs=system_IDs,
    #         countries=countries,
    #         include_resource_recovery=True,
    #         load_cached_data=True,
    #         )