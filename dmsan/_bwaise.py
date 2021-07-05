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
from dmsan import data_path, results_path, AHP, MCDA

__all__ = ('baseline_tech_scores', )
data_path_tech_scores = os.path.join(data_path, 'technology_scores.xlsx')

# Util function
tech_file = pd.ExcelFile(data_path_tech_scores)
read_excel = lambda name: pd.read_excel(tech_file, name).expected


# %%

# =============================================================================
# Technology scores
# =============================================================================

def check_lca(lca_perspective):
    if not lca_perspective.upper() in ('I', 'H', 'E'):
        raise ValueError('`lca_perspective` can only be "I", "H", or "E", '
                         f'not "{lca_perspective}".')


# Baseline
def get_baseline_tech_scores(lca_perspective='H'):
    check_lca(lca_perspective)

    # Technical
    tech_score_T_All = pd.DataFrame([
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

    tech_score_T_All.columns = [f'T{i+1}' for i in range(tech_score_T_All.shape[1])]

    # Resource Recovery
    # Import simulated results
    baseline = pd.read_csv(os.path.join(data_path, 'bwaise_baseline.tsv'),
                           index_col=(0, 1), sep='\t')

    tech_score_RR_All = pd.DataFrame([
        read_excel('water_reuse'),
        baseline.loc[('Net recovery', 'N')].values,
        baseline.loc[('Net recovery', 'P')].values,
        baseline.loc[('Net recovery', 'K')].values,
        baseline.loc[('Net recovery', 'energy')].values,
        read_excel('supply_chain')
        ]).transpose()

    tech_score_RR_All.columns = [f'RR{i+1}' for i in range(tech_score_RR_All.shape[1])]

    # Economic
    tech_score_Econ_All = pd.DataFrame([
        baseline.loc[('TEA results', 'Net cost')].values
        ]).transpose()

    tech_score_Econ_All.columns = ['Econ1']

    # Environmental, lca_perspective can be "I", "H", or "E" for
    # individualist, hierarchist, or egalitarian
    lca_ind = [ind for ind in baseline.index if ind[1].startswith(f'{lca_perspective.upper()}_')]
    tech_score_Env_All = pd.DataFrame([
        baseline[baseline.index==lca_ind[0]].values[0], # ecosystem quality
        baseline[baseline.index==lca_ind[1]].values[0], # human health
        baseline[baseline.index==lca_ind[2]].values[0], # resource depletion
        ]).transpose()

    tech_score_Env_All.columns = [f'Env{i+1}' for i in range(tech_score_Env_All.shape[1])]

    # Social
    tech_score_S_All = pd.DataFrame([
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

    tech_score_S_All.columns = [f'S{i+1}' for i in range(tech_score_S_All.shape[1])]

    # `tech_scores` is `Tech_Scores_compiled` in the original script,
    # values checked to be the same as the original script
    tech_scores = pd.concat([tech_score_T_All, tech_score_RR_All, tech_score_Econ_All,
                             tech_score_Env_All, tech_score_S_All], axis=1)

    return tech_scores

baseline_tech_scores = get_baseline_tech_scores()


# With uncertainties (only some certain parameters are affected)
varied_inds = [*[f'RR{i}' for i in range(2, 6)],
               'Econ1',
               *[f'Env{i}' for i in range(1, 4)]]
# varied_idx = baseline_tech_scores.columns.get_indexer_for(varied_inds)

def get_uncertainty_scores(lca_perspective='H', baseline_scores=None):
    check_lca(lca_perspective)

    if not baseline_scores:
        baseline_scores = get_baseline_tech_scores()

    path = os.path.join(data_path, 'bwaise_uncertainties.xlsx')
    file = pd.ExcelFile(path)
    sysA = pd.read_excel(file, 'sysA', index_col=0, header=(0, 1))
    sysB = pd.read_excel(file, 'sysB', index_col=0, header=(0, 1))
    sysC = pd.read_excel(file, 'sysC', index_col=0, header=(0, 1))

    col_names = [
        ('N recovery', 'Total N'),
        ('P recovery', 'Total P'),
        ('K recovery', 'Total K'),
        ('COD recovery', 'Total COD'),
        ('TEA results', 'Annual net cost [USD/cap/yr]'),
        ('LCA results', f'Net emission {lca_perspective.upper()}_EcosystemQuality_Total [points/cap/yr]'),
        ('LCA results', f'Net emission {lca_perspective.upper()}_HumanHealth_Total [points/cap/yr]'),
        ('LCA results', f'Net emission {lca_perspective.upper()}_Resources_Total [points/cap/yr]')
        ]

    sysA_val = sysA[col_names]
    sysB_val = sysB[col_names]
    sysC_val = sysC[col_names]

    tech_score_dct = {}
    N = sysA.shape[0]
    for i in range(N):
        simulated = pd.DataFrame([sysA_val.iloc[i],
                                  sysB_val.iloc[i],
                                  sysC_val.iloc[i]]).reset_index(drop=True)
        if simulated.isna().values.any(): # failed evaluation, drop it
            continue

        tech_scores = baseline_scores.copy()
        tech_scores[varied_inds] = simulated
        tech_score_dct[i] = tech_scores

    return tech_score_dct

uncertainty_tech_scores = get_uncertainty_scores()


# %%

# =============================================================================
# MCDA using TOPSIS
# =============================================================================

# Names of the alternative systems
alt_names = pd.read_excel(data_path_tech_scores, sheet_name='user_interface').system

# Baseline
# `bwaise_ahp.norm_weights_df` is `subcriteria_weights` in the original script,
# values checked to be the same as the original script
bwaise_ahp = AHP(location_name='Uganda', num_alt=len(alt_names),
                 na_default=0.00001, random_index={})

# `bwaise_mcda.score` is `performance_score_FINAL` in the original script,
# `bwaise_mcda.rank` is `ranking_FINAL` in the original script,
# values checked to be the same as the original script
# Note that the small discrepancies in scores are due to the rounding error
# in the original script (weights of 0.34, 0.33, 0.33 instead of 1/3 for Env)
bwaise_mcda = MCDA(method='TOPSIS', alt_names=alt_names,
                   indicator_weights=bwaise_ahp.norm_weights_df,
                   tech_scores=baseline_tech_scores)
bwaise_mcda.run_MCDA()
# If want to export the results
path = os.path.join(results_path, 'RESULTS_AHP_TOPSIS.xlsx')
# with pd.ExcelWriter(path) as writer:
#     bwaise_mcda.score.to_excel(writer, sheet_name='Score')
#     bwaise_mcda.rank.to_excel(writer, sheet_name='Rank')

# Uncertainties
def run_uncertainty_mcda(mcda, criteria_weights, tech_score_dct):
    scores = []
    ranks = []

    for k, v in tech_score_dct.items():
        mcda.scores = v
        mcda.run_MCDA(criteria_weights=criteria_weights)
        scores.append(mcda.score)
        ranks.append(mcda.rank)

    scores_df = pd.concat(scores)
    ranks_df = pd.concat(ranks)

    return scores_df, ranks_df

uncertainty_perform_scores, uncertainty_ranks = \
    run_uncertainty_mcda(bwaise_mcda, bwaise_mcda.criteria_weights.iloc[-1], # 1:1:1:1:1
                         uncertainty_tech_scores)

# If want to export the results
path = os.path.join(results_path, 'RESULTS_AHP_TOPSIS_uncertainties.xlsx')
# with pd.ExcelWriter(path) as writer:
#     uncertainty_perform_scores.to_excel(writer, sheet_name='Score')
#     uncertainty_ranks.to_excel(writer, sheet_name='Rank')


# %%

# =============================================================================
#
# =============================================================================