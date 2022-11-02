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

import os, numpy as np, pandas as pd
from qsdsan.utils import time_printer
from dmsan.utils import get_uncertainties
from dmsan.comparison import scores_path, simulate_models, get_models

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
N_across = 10
N_step = 1
N_no_fertilizer = 20
seed = 3221

def get_param(model, name):
    names = (name,) if isinstance(name, str) else name
    for name in names: # just for the sake of operator daily wage(s)
        for i in model.parameters:
            if i.name == name: return i
    raise ValueError(f'Cannot find parameter "{name}".')


# %%

def export_percentiles(uncertainty_dct, q=[0.05, 0.25, 0.5, 0.75, 0.95], path=''):
    percentiles = {}
    for key, df in uncertainty_dct.items():
        module, country, kind = key.split('_')
        if kind == 'params': continue
        percentiles[f'{module}_{country}'] = df.quantile(q=q)

    percentile_df = pd.concat(percentiles.values())
    percentile_df.index = pd.MultiIndex.from_product([percentiles.keys(), q], names=['module', 'percentile'])
    percentile_df = percentile_df.unstack()

    if path is not None:
        path = path or os.path.join(scores_path, 'percentiles.xlsx')
        percentile_df.to_excel(path)

    return percentile_df


# %%

factor_vals = np.arange(0, 1.1, 0.5) # start, stop, step (stop is excluded)
@time_printer
def evaluate_across_price_factor(model_dct, N=N_across, seed=seed, vals=factor_vals):
    dct = {}
    for val in vals:
        print(f'\n\nprice factor: {val}')
        model_dct_new = {}
        for key, model_original in model_dct.items():
            price_factor = get_param(model_original, 'Price factor')
            model_new = model_original.copy()
            model_new.parameters = [p for p in model_original.parameters if p is not price_factor]
            price_factor.setter(val)
            model_dct_new[key] = model_new

        uncertinty_dct = get_uncertainties(model_dct=model_dct_new, N=N_across, print_time=False)
        dct[val] = export_percentiles(uncertinty_dct, path=None)
    price_factor_path = os.path.join(scores_path, 'price_factor_percentiles.xlsx')
    writer = pd.ExcelWriter(price_factor_path)
    for name, df in dct.items():
        df.to_excel(writer, sheet_name=str(name))
    writer.save()

    return dct


# %%

br_wage_vals = np.linspace(1.0015, 336.9715, num=N_step+2)
ngre_wage_vals = np.linspace(0.125188, 42.12144, num=N_step+2)

@time_printer
def evaluate_across_wages(model_dct, N=N_across, seed=seed):
    dct = {}
    
    for n in range(N_step+2):
        print(f'\n\n{n} step in wage list')
        model_dct_new = {}
        for key, model_original in model_dct.items():
            if key[:2] == 'br':
                wage_param_name = ('Operator daily wages', 'Operator daily wage')
                val = br_wage_vals[n]
            else:
                wage_param_name = ('Wages', 'Labor wages')
                val = ngre_wage_vals[n]
            wage_param = get_param(model_original, wage_param_name)
            model_new = model_original.copy()
            model_new.parameters = [p for p in model_original.parameters if p is not wage_param]
            wage_param.setter(val)
            model_dct_new[key] = model_new

        uncertinty_dct = get_uncertainties(model_dct=model_dct_new, N=N_across, print_time=False)
        dct[n] = export_percentiles(uncertinty_dct, path=None)
    wages_path = os.path.join(scores_path, 'wages_percentiles.xlsx')
    writer = pd.ExcelWriter(wages_path)
    for name, df in dct.items():
        df.to_excel(writer, sheet_name=str(name))
    writer.save()

    return dct


# %%

@time_printer
def evaluate_without_fertilizer_recovery(model_dct, N=N_across, seed=seed):
    dct = {}
    print('\n\nno fertilizer recovery')
    for key, model_original in model_dct.items():
        fertilizer_params = []
        for p in model_original.parameters:
            if 'fertilizer' in p.name:
                if 'CF' in p.name or 'price' in p.name: fertilizer_params.append(p)

        # # If want to know what's being set to 0
        # for p in fertilizer_params: print(p.name)

        model_new = model_original.copy()
        model_new.parameters = [p for p in model_original.parameters if p not in fertilizer_params]
        for p in fertilizer_params: p.setter(0)
        dct[key] = model_new

    uncertinty_dct = get_uncertainties(model_dct=dct, N=N_no_fertilizer, print_time=False)
    path = os.path.join(scores_path, 'no_fertilizer_percentiles.xlsx')
    df = export_percentiles(uncertinty_dct, path=path)

    return df

# %%

if __name__ == '__main__':
    # outs = simulate_models(
    #     countries=countries, N=N, seed=seed, 
    #     include_resource_recovery=False,
    #     include_general_model=True,
    #     include_baseline=True,
    #     include_spearman=False,
    #     pickle_path='',
    #     )
    # baseline_df, uncertainty_dct, model_dct = outs
    
    model_dct = simulate_models(
        countries=countries, N=N, seed=seed, 
        include_resource_recovery=False,
        include_general_model=True,
        include_baseline=True,
        include_spearman=False,
        pickle_path='',
        skip_evaluation=True
        )
    wage_dct = evaluate_across_wages(model_dct)
    
    # percentile_df = export_percentiles(uncertainty_dct)
    # price_factor_dct = evaluate_across_price_factor(model_dct)
    # fertilizer_df = evaluate_without_fertilizer_recovery(model_dct)