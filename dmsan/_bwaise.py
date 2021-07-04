#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 17:22:03 2021

@authors:
    Tori Morgan <vlmorgan@illinois.edu>,
    Hannah Lohman <hlohman94@gmail.com>,
    Stetson Rowles <stetsonsc@gmail.com>,
    Yalin Li <zoe.yalin.li@gmail.com>
"""

import os
import pandas as pd

# __all__ = ('get_baseline', 'save_baseline', 'get_uncertainties', 'save_uncertainties')

data_path = os.path.join(os.path.dirname(__file__), 'data')

data_path_tech_scores = os.path.join(data_path, 'technology_scores.xlsx')

tech_file = pd.ExcileFile(data_path_tech_scores)
# Util function
read_excel = lambda name: pd.read_excel(tech_file, name).expected


# %%

# =============================================================================
# Technology scores
# =============================================================================

# Technical
Tech_Score_T_All = pd.DataFrame([
    read_excel('user_interface'),
    read_excel('treatment_type'),
    read_excel('system_part_accessibility'),
    read_excel('design_transport'),
    read_excel('construction_skills'),
    read_excel('OM_complexity'),
    read_excel('pop_flexibility'),
    read_excel('electricity_flexibility'),
    read_excel('drought_flexibility')
    ]).transpose()

Tech_Score_T_All.columns = [f'T{i+1}' for i in range(Tech_Score_T_All.shape[0])]

# Resource Recovery
# Import simulated results
baseline = pd.read_csv(os.path.join(data_path, 'bwaise_baseline.tsv'),
                       index_col=(0, 1), sep='\t')

Tech_Score_RR_All = pd.DataFrame([
    read_excel('water_reuse'),
    baseline.loc[('Net recovery', 'N')].values,
    baseline.loc[('Net recovery', 'P')].values,
    baseline.loc[('Net recovery', 'K')].values,
    baseline.loc[('Net recovery', 'energy')].values,
    read_excel('supply_chain')
    ]).transpose()

Tech_Score_RR_All.columns = [f'RR{i+1}' for i in range(Tech_Score_RR_All.shape[0])]

# Economic
Tech_Score_Econ_All = pd.DataFrame([
    baseline.loc[('TEA results', 'Net cost')].values
    ]).transpose()

Tech_Score_Econ_All.columns = ['Econ1']

# Environmental
H_ind = [ind for ind in baseline.index if ind[1].startswith('H_')] # hierarchist
Tech_Score_Env_All = pd.DataFrame([
    baseline[baseline.index==H_ind[0]].values[0], # ecosystem quality
    baseline[baseline.index==H_ind[1]].values[0], # human health
    baseline[baseline.index==H_ind[2]].values[0], # resource depletion
    ]).transpose()

Tech_Score_Env_All.columns = [f'Env{i+1}' for i in range(Tech_Score_Env_All.shape[0])]


# Social
Tech_Score_S_All = pd.DataFrame([
    read_excel('design_job_creation'),
    read_excel('design_high_pay_jobs'),
    read_excel('end_user_disposal'),
    read_excel('end_user_cleaning'),
    read_excel('privacy'),
    read_excel('odor'),
    read_excel('security'),
    read_excel('management_disposal'),
    read_excel('management_cleaning')
    ]).transpose()

Tech_Score_S_All.columns = [f'S{i+1}' for i in range(Tech_Score_S_All.shape[0])]

# all_scores = [Tech_Score_T_All, Tech_Score_RR_All, Tech_Score_S_All]

# for i in all_scores:
#     i.columns =