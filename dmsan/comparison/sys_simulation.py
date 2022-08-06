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
baseline_dct = dict.fromkeys(countries)
uncertainty_dct = dict.fromkeys(countries)

# # If want to ensure consistent parameter values across the different systems
# # for a particular country
# from dmsan.biogenic_refinery import get_models
# model_dct = get_models(system_IDs=system_IDs, countries=countries,load_cached_data=False,)

if __name__ == '__main__':
    for country in countries:
        baseline_dct[country], uncertainty_dct[country] = simulate_models(country, N=N, seed=seed)