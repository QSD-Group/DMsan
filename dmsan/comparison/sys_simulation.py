#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making for sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>

Run this module to save the results to the /results folder to avoid repeating
simulating the system.
'''

import os
from dmsan.comparison import (
    scores_path,
    get_baseline,
    get_models,
    get_uncertainties,
    )

# Comment these out if want to see all warnings
import warnings
warnings.filterwarnings(action='ignore')


# %%

# =============================================================================
# Run all simulations
# =============================================================================

def run_simulations(country, N, seed=None):
    global baseline_df, uncertainty_dct
    country_folder = os.path.join(scores_path, country)
    # Create the folder if there isn't one already
    if not os.path.isdir(country_folder): os.mkdir(country_folder)
    baseline_path = os.path.join(country_folder, 'sys_baseline.tsv')
    param_path = os.path.join(country_folder, 'parameters.xlsx')
    pickle_path = os.path.join(country_folder, 'model_data.pckl')
    uncertainty_path = os.path.join(country_folder, 'sys_uncertainties.xlsx')

    models = get_models(country=country, load_cached_data=False)
    baseline_df = get_baseline(models=models, file_path=baseline_path)
    uncertainty_dct = get_uncertainties(country=country, N=N, seed=seed,
                                        param_path=param_path,
                                        pickle_path=pickle_path,
                                        result_path=uncertainty_path)


if __name__ == '__main__':
    for country in ('China', 'India', 'Senegal', 'South Africa', 'Uganda'):
        run_simulations(country, N=20, seed=3221)