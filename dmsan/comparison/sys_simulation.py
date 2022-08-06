#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making of sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>

Run this module to save the results to the /results folder to avoid repeating
simulating the system.
'''

from dmsan.comparison import simulate_models

# Comment these out if want to see all warnings
import warnings
warnings.filterwarnings(action='ignore')

countries = ('China', 'India', 'Senegal', 'South Africa', 'Uganda')
N = 20
seed = 3221

if __name__ == '__main__':
    baseline_df, uncertainty_dct = simulate_models(countries=countries, N=N, seed=seed)

    # # To reload models
    # from dmsan.comparison import get_models
    # model_dct = get_models(
    #         countries=countries,
    #         load_cached_data=True,
    #         )