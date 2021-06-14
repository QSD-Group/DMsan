#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 13:31:44 2021

@author: Yalin Li <zoe.yalin.li@gmail.com>

"""

'''
TODO: figure out why ReCiPe gives so much credit to fertilizers
'''

# Run this module to save the results to the /data folder to avoid repeating
# simulating the system

import os
import pandas as pd
from exposan import bwaise as bw

__all__ = ('get_lca_baseline', 'get_lca_uncertainties')

data_path = os.path.join(os.path.dirname(__file__), 'data')


# %%

# =============================================================================
# Baseline for Individualist, Hierarchist, and Egalitarian (I/H/E)
# =============================================================================

def update_lca():
    bw.update_lca_data('new')
    lcas = lcaA, lcaB, lcaC = bw.lcaA, bw.lcaB, bw.lcaC

    indicators = []
    for cat in ('I', 'H', 'E'):
        temp = [i for i in lcaA.indicators if cat+'_' in i.ID and '_Total' in i.ID]
        temp.sort(key=lambda ind: ind.ID)
        indicators.extend(temp)

    for lca in lcas:
        lca.indicators = indicators

    return lcas


def get_cap_yr_pts(lca):
    impact_dct = lca.get_total_impacts()
    ratio = lca.lifetime * bw.systems.get_ppl(lca.system.ID[-1])
    for k, v in impact_dct.items():
        impact_dct[k] = v / ratio
    return impact_dct


def get_lca_baseline():
    lcas = update_lca()
    baseline_dct = {}

    for lca in lcas:
        baseline_dct[lca.system.ID] = get_cap_yr_pts(lca)

    df = pd.DataFrame.from_dict(baseline_dct)
    return df

def save_lca_baseline(path=''):
    if not path:
        path = os.path.join(data_path, 'lca_baseline.tsv')
    sep = '\t' if path.endswith('.tsv') else ''
    lca_baseline_df = get_lca_baseline()
    lca_baseline_df.to_csv(os.path.join(data_path, 'lca_baseline.tsv'), sep=sep)
    return lca_baseline_df

lca_baseline_df = save_lca_baseline()


# %%

# =============================================================================
# Add uncertainties
# =============================================================================

def get_lca_uncertainties():
    from exposan.bwaise.models import update_metrics, update_LCA_CF_parameters, run_uncertainty
    models = modelA, modelB, modelC = bw.modelA, bw.modelB, bw.modelC

    for model in models:
        model = update_LCA_CF_parameters(model, 'new')
        model = update_metrics(model, 'new')

    uncertainty_dct = {}
    for model in models:
        uncertainty_dct[model.system.ID] = run_uncertainty(model, N=10)['data']
    return uncertainty_dct

def save_lca_uncertainties(path=''):
    if not path:
        path = os.path.join(data_path, 'lca_uncertainties.xlsx')
    lca_uncertainty_dct = get_lca_uncertainties()
    writer = pd.ExcelWriter(path)

    for sys_ID, df in lca_uncertainty_dct.items():
        lca_columns = [col for col in df.columns if 'net emission' in col[-1].lower()]
        new_df = df[lca_columns]
        new_df.to_excel(writer, sheet_name=sys_ID)
    writer.save()

    return lca_uncertainty_dct

lca_uncertainty_dct = save_lca_uncertainties()