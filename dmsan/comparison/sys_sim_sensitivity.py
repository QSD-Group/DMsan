#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making of sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/DMsan/blob/main/LICENSE.txt
for license details.

Run this module to save the results to the /scores folder to avoid repeating
simulating the system.
'''

import os
from dmsan.comparison import scores_path, simulate_models

# Comment these out if want to see all warnings
import warnings
warnings.filterwarnings(action='ignore')

# 77 countries broken down into groups of 5
countries = ('Albania',)
# countries = ('Albania', 'Armenia', 'Austria', 'Bangladesh', 'Barbados')
# countries = ('Belarus', 'Belgium', 'Belize', 'Bolivia', 'Botswana')
# countries = ('Bulgaria', 'Cambodia', 'Cameroon', 'Chile', 'China')
# countries = ('Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Dominican Republic')
# countries = ('Ecuador', 'Egypt', 'El Salvador', 'Estonia', 'Finland')
# countries = ('France', 'Georgia', 'Germany', 'Ghana', 'Greece')
# countries = ('Guatemala', 'Guyana', 'Honduras', 'Hungary', 'India')
# countries = ('Israel', 'Italy', 'Jordan', 'Kazakhstan', 'Kenya')
# countries = ('South Korea', 'Kyrgyzstan', 'Latvia', 'Lebanon', 'Lithuania')
# countries = ('Luxembourg', 'Malaysia', 'Mauritius', 'Moldova', 'Mongolia')
# countries = ('Montenegro', 'Netherlands', 'New Zealand', 'Nigeria', 'Norway')
# countries = ('Pakistan', 'Philippines', 'Poland', 'Portugal', 'Romania')
# countries = ('Russia', 'Rwanda', 'Saudi Arabia', 'Slovakia', 'Slovenia')
# countries = ('Spain', 'Sri Lanka', 'Sweden', 'Switzerland', 'Tanzania')
# countries = ('Thailand', 'Turkey', 'Uganda', 'Ukraine', 'United Kingdom', 'United States', 'Uruguay')

# All 77 countries
# countries = ('Albania', 'Armenia', 'Austria', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Bolivia',
#              'Botswana', 'Bulgaria', 'Cambodia', 'Cameroon', 'Chile', 'China', 'Croatia', 'Cyprus', 'Czech Republic',
#              'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Estonia', 'Finland', 'France',
#              'Georgia', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Guyana', 'Honduras', 'Hungary', 'India', 'Israel',
#              'Italy', 'Jordan', 'Kazakhstan', 'Kenya', 'South Korea', 'Kyrgyzstan', 'Latvia', 'Lebanon', 'Lithuania',
#              'Luxembourg', 'Malaysia', 'Mauritius', 'Moldova', 'Mongolia', 'Montenegro', 'Netherlands', 'New Zealand',
#              'Nigeria', 'Norway', 'Pakistan', 'Philippines', 'Poland', 'Portugal', 'Romania', 'Russia', 'Rwanda',
#              'Saudi Arabia', 'Slovakia', 'Slovenia', 'Spain', 'Sri Lanka', 'Sweden', 'Switzerland', 'Tanzania',
#              'Thailand', 'Turkey', 'Uganda', 'Ukraine', 'United Kingdom', 'United States', 'Uruguay')


N = 100
seed = 3221

outs = simulate_models(
    countries=countries, N=N, seed=seed, 
    include_resource_recovery=False,
    include_general_model=True,
    country_specific_model_kwargs={'include_non_contextual_params': False},
    include_baseline=True,
    include_spearman=True,
    baseline_path=os.path.join(scores_path, 'simulated_baseline_sensitivity.csv'),
    uncertainty_path=os.path.join(scores_path, 'simulated_uncertainties_sensitivity.xlsx'),
    spearman_path_prefix=os.path.join(scores_path, 'simulated_spearman'),
    )
baseline_df, uncertainty_dct, model_dct = outs