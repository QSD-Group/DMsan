#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making of sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>

    Hannah Lohman <hlohman94@gmail.com>

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
    baseline_df, uncertainty_dct = simulate_models(
        system_IDs=system_IDs, countries=countries, N=N, seed=seed)

    # # To reload models
    # model_dct = get_models(
    #         system_IDs=system_IDs,
    #         countries=countries,
    #         load_cached_data=True,
    #         )