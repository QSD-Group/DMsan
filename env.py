#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 13:31:44 2021

@author: Yalin Li <zoe.yalin.li@gmail.com>

TODO: figure out why ReCiPe gives so much credit to fertilizers

"""

# %%

# =============================================================================
# Baseline for Individualist, Hierarchist, and Egalitarian (I/H/E)
# =============================================================================

from exposan import bwaise as bw
bw.update_lca_data('new')

lcas = lcaA, lcaB, lcaC = bw.lcaA, bw.lcaB, bw.lcaC
get_ppl = bw.systems.get_ppl

indicators = []
for cat in ('I', 'H', 'E'):
    temp = [i for i in lcaA.indicators if cat+'_' in i.ID and '_Total' in i.ID]
    temp.sort(key=lambda ind: ind.ID)
    indicators.extend(temp)

for lca in lcas:
    lca.indicators = indicators


def get_cap_yr_pts(lca):
    impact_dct = lca.get_total_impacts()
    ratio = lca.lifetime * get_ppl(lca.system.ID[-1])
    for k, v in impact_dct.items():
        impact_dct[k] = v / ratio
    return impact_dct


# To get some results
baseline_dct = {}
for lca in lcas:
    baseline_dct[lca.system.ID] = get_cap_yr_pts(lca)


# %%

# =============================================================================
# Add uncertainties
# =============================================================================

from exposan.bwaise.models import update_metrics, update_LCA_CF_parameters, run_uncertainty
models = modelA, modelB, modelC = bw.modelA, bw.modelB, bw.modelC

for model in models:
    model = update_LCA_CF_parameters(model, 'new')
    model = update_metrics(model, 'new')

uncertainty_dct = {}
for model in models:
    uncertainty_dct[model.system.ID] = run_uncertainty(model, N=10)