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

import os, numpy as np, pandas as pd
from chaospy import distributions as shape
from qsdsan.utils import time_printer
from dmsan.utils import get_uncertainties
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


N = 10
N_across = 50
N_step = 5  #5
N_no_fertilizer = 20
seed = 3221

def get_param(model, name):
    names = (name,) if isinstance(name, str) else name
    for name in names: # just for the sake of operator daily wage(s)
        for i in model.parameters:
            if i.name == name: return i
    raise ValueError(f'Cannot find parameter "{name}".')


# Set a very tight distribution to approximate a constant
def add_constant_param(model, parameter, constant):
    if constant > 0:
        D = shape.Uniform(lower=constant*(1-10**(-6)), upper=constant*(1+10**(-6)))
    elif constant < 0:
        D = shape.Uniform(lower=constant*(1+10**(-6)), upper=constant*(1-10**(-6)))
    else:
        D = shape.Uniform(lower=-10**(-6), upper=10**(-6))
    model.parameter(
        setter=parameter.setter,
        name=parameter.name,
        element=parameter.element,
        kind=parameter.kind,
        units=parameter.units,
        baseline=constant,
        distribution=D,
        )


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

# Wages and Operator daily wages
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
            add_constant_param(model_new, wage_param, val)
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

# Price level ratio
price_ratio_vals = np.linspace(0.174, 1.370785956, num=N_step+2)
@time_printer
def evaluate_across_price_ratio(model_dct, N=N_across, seed=seed, vals=price_ratio_vals):
    dct = {}
    for val in vals:
        print(f'\n\nprice ratio: {val}')
        model_dct_new = {}
        for key, model_original in model_dct.items():
            price_ratio_param = get_param(model_original, 'Price ratio')
            model_new = model_original.copy()
            model_new.parameters = [p for p in model_original.parameters if p is not price_ratio_param]
            add_constant_param(model_new, price_ratio_param, val)
            model_dct_new[key] = model_new

        uncertinty_dct = get_uncertainties(model_dct=model_dct_new, N=N_across, print_time=False)
        dct[val] = export_percentiles(uncertinty_dct, path=None)
    price_ratio_path = os.path.join(scores_path, 'price_ratio_percentiles.xlsx')
    writer = pd.ExcelWriter(price_ratio_path)
    for name, df in dct.items():
        df.to_excel(writer, sheet_name=str(name))
    writer.save()

    return dct


# %%

# Electricity price
electricity_price_vals = np.linspace(0.003, 0.378, num=N_step+2)
@time_printer
def evaluate_across_electricity_price(model_dct, N=N_across, seed=seed, vals=electricity_price_vals):
    dct = {}
    for val in vals:
        print(f'\n\nelectricity price: {val}')
        model_dct_new = {}
        for key, model_original in model_dct.items():
            electricity_price_param_name = ('Energy price', 'Electricity price')
            electricity_price_param = get_param(model_original, electricity_price_param_name)
            model_new = model_original.copy()
            model_new.parameters = [p for p in model_original.parameters if p is not electricity_price_param]
            add_constant_param(model_new, electricity_price_param, val)
            model_dct_new[key] = model_new

        uncertinty_dct = get_uncertainties(model_dct=model_dct_new, N=N_across, print_time=False)
        dct[val] = export_percentiles(uncertinty_dct, path=None)
    electricity_price_path = os.path.join(scores_path, 'electricity_price_percentiles.xlsx')
    writer = pd.ExcelWriter(electricity_price_path)
    for name, df in dct.items():
        df.to_excel(writer, sheet_name=str(name))
    writer.save()

    return dct


# %%

# Electricity GWP
electricity_gwp_vals = np.linspace(0.012, 1.046968, num=N_step+2)
ngre_solar_gwp_vals = np.linspace(0.0, 0.0, num=N_step+2)
@time_printer
def evaluate_across_electricity_gwp(model_dct, N=N_across, seed=seed):
    dct = {}
    for n in range(N_step+2):
        print(f'\n\n{n} step in electricity gwp list')
        model_dct_new = {}
        for key, model_original in model_dct.items():
            electricity_gwp_param_name = ('Electricity CF', 'Energy gwp')
            electricity_gwp_param = get_param(model_original, electricity_gwp_param_name)
            model_new = model_original.copy()
            model_new.parameters = [p for p in model_original.parameters if p is not electricity_gwp_param]

            if key[:3] == 'ngA':
                val = ngre_solar_gwp_vals[n]
            elif key[:3] == 'reC':
                val = ngre_solar_gwp_vals[n]
            else:
                val = electricity_gwp_vals[n]
            add_constant_param(model_new, electricity_gwp_param, val)
            model_dct_new[key] = model_new

        uncertinty_dct = get_uncertainties(model_dct=model_dct_new, N=N_across, print_time=False)
        dct[n] = export_percentiles(uncertinty_dct, path=None)
    electricity_gwp_path = os.path.join(scores_path, 'electricity_gwp_percentiles.xlsx')
    writer = pd.ExcelWriter(electricity_gwp_path)
    for name, df in dct.items():
        df.to_excel(writer, sheet_name=str(name))
    writer.save()

    return dct


# %%

# E cal
e_cal_vals = np.linspace(1786, 3885, num=N_step+2)
@time_printer
def evaluate_across_e_cal(model_dct, N=N_across, seed=seed, vals=e_cal_vals):
    dct = {}
    for val in vals:
        print(f'\n\ne cal: {val}')
        model_dct_new = {}
        for key, model_original in model_dct.items():
            e_cal_param_name = ('Excretion e cal', 'E cal')
            e_cal_param = get_param(model_original, e_cal_param_name)
            model_new = model_original.copy()
            model_new.parameters = [p for p in model_original.parameters if p is not e_cal_param]
            add_constant_param(model_new, e_cal_param, val)
            model_dct_new[key] = model_new
            
        uncertinty_dct = get_uncertainties(model_dct=model_dct_new, N=N_across, print_time=False)
        dct[val] = export_percentiles(uncertinty_dct, path=None)
    e_cal_path = os.path.join(scores_path, 'e_cal_percentiles.xlsx')
    writer = pd.ExcelWriter(e_cal_path)
    for name, df in dct.items():
        df.to_excel(writer, sheet_name=str(name))
    writer.save()

    return dct


# %%

# P anim
p_anim_vals = np.linspace(6.55, 104.98, num=N_step+2)
@time_printer
def evaluate_across_p_anim(model_dct, N=N_across, seed=seed, vals=p_anim_vals):
    dct = {}
    for val in vals:
        print(f'\n\np anim: {val}')
        model_dct_new = {}
        for key, model_original in model_dct.items():
            p_anim_param_name = ('Excretion p anim', 'P anim')
            p_anim_param = get_param(model_original, p_anim_param_name)
            model_new = model_original.copy()
            model_new.parameters = [p for p in model_original.parameters if p is not p_anim_param]
            add_constant_param(model_new, p_anim_param, val)
            model_dct_new[key] = model_new

        uncertinty_dct = get_uncertainties(model_dct=model_dct_new, N=N_across, print_time=False)
        dct[val] = export_percentiles(uncertinty_dct, path=None)
    p_anim_path = os.path.join(scores_path, 'p_anim_percentiles.xlsx')
    writer = pd.ExcelWriter(p_anim_path)
    for name, df in dct.items():
        df.to_excel(writer, sheet_name=str(name))
    writer.save()

    return dct


# %%

# P veg
p_veg_vals = np.linspace(24.81, 73.29, num=N_step+2)
@time_printer
def evaluate_across_p_veg(model_dct, N=N_across, seed=seed, vals=p_veg_vals):
    dct = {}
    for val in vals:
        print(f'\n\np_veg: {val}')
        model_dct_new = {}
        for key, model_original in model_dct.items():
            p_veg_param_name = ('Excretion p veg', 'P veg')
            p_veg_param = get_param(model_original, p_veg_param_name)
            model_new = model_original.copy()
            model_new.parameters = [p for p in model_original.parameters if p is not p_veg_param]
            add_constant_param(model_new, p_veg_param, val)
            model_dct_new[key] = model_new

        uncertinty_dct = get_uncertainties(model_dct=model_dct_new, N=N_across, print_time=False)
        dct[val] = export_percentiles(uncertinty_dct, path=None)
    p_veg_path = os.path.join(scores_path, 'p_veg_percentiles.xlsx')
    writer = pd.ExcelWriter(p_veg_path)
    for name, df in dct.items():
        df.to_excel(writer, sheet_name=str(name))
    writer.save()

    return dct


# %%

factor_vals = np.arange(0, 1.1, 0.5) # start, stop, step (stop is excluded)
@time_printer
def evaluate_across_price_factor(model_dct, N=N_across, seed=seed, vals=factor_vals):
    dct = {}
    for val in vals:
        print(f'\n\nprice factor: {val}')
        model_dct_new = {}
        for key, model_original in model_dct.items():
            price_factor_param = get_param(model_original, 'Price factor')
            model_new = model_original.copy()
            model_new.parameters = [p for p in model_original.parameters if p is not price_factor_param]
            add_constant_param(model_new, price_factor_param, val)
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
    # wage_dct = evaluate_across_wages(model_dct)
    # price_ratio_dct = evaluate_across_price_ratio(model_dct)
    # electricity_price_dct = evaluate_across_electricity_price(model_dct)
    electricity_gwp_dct = evaluate_across_electricity_gwp(model_dct)
    # e_cal_dct = evaluate_across_e_cal(model_dct)
    # p_anim_dct = evaluate_across_p_anim(model_dct)
    # p_veg_dct = evaluate_across_p_veg(model_dct)
    
    # percentile_df = export_percentiles(uncertainty_dct)
    # price_factor_dct = evaluate_across_price_factor(model_dct)
    # fertilizer_df = evaluate_without_fertilizer_recovery(model_dct)