#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making of sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>

    Hannah Lohman <hlohman94@gmail.com>

Run this module to save the results to the /results folder to avoid repeating
simulating the system.
'''

from dmsan.new_generator import simulate_models

# Comment these out if want to see all warnings
import warnings
warnings.filterwarnings(action='ignore')

countries = ('China', 'India', 'Senegal', 'South Africa', 'Uganda')
N = 20
seed = 3221
baseline_dct = dict.fromkeys(countries)
uncertainty_dct = dict.fromkeys(countries)

if __name__ == '__main__':
    for country in countries:
        baseline_dct[country], uncertainty_dct[country] = simulate_models(country, N=N, seed=seed)