#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 17:22:03 2021

@authors:
    Tori Morgan <vlmorgan@illinois.edu>,
    Hannah Lohman <hlohman94@gmail.com>,
    Stetson Rowles <stetsonsc@gmail.com>,
    Yalin Li <zoe.yalin.li@gmail.com>

Part of this module is based on the BioSTEAM and QSD packages:
https://github.com/BioSTEAMDevelopmentGroup/biosteam
https://github.com/QSD-group/QSDsan
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
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

def get_uncertainty_data(lca_perspective='H', baseline_scores=None):
    check_lca(lca_perspective)

    if not baseline_scores:
        baseline_scores = get_baseline_tech_scores()

    file_path = os.path.join(data_path, 'bwaise_uncertainties.xlsx')
    file = pd.ExcelFile(file_path)
    paramA = pd.read_excel(file, 'sysA-param', index_col=0, header=(0, 1))
    paramB = pd.read_excel(file, 'sysB-param', index_col=0, header=(0, 1))
    paramC = pd.read_excel(file, 'sysC-param', index_col=0, header=(0, 1))
    param_dct = dict(sysA=paramA, sysB=paramB, sysC=paramC)

    sysA = pd.read_excel(file, 'sysA-results', index_col=0, header=(0, 1))
    sysB = pd.read_excel(file, 'sysB-results', index_col=0, header=(0, 1))
    sysC = pd.read_excel(file, 'sysC-results', index_col=0, header=(0, 1))

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

        tech_scores = baseline_scores.copy()
        tech_scores[varied_inds] = simulated
        tech_score_dct[i] = tech_scores

    return param_dct, tech_score_dct

param_dct, tech_score_dct = get_uncertainty_data()


# %%

# =============================================================================
# TOPSIS baseline for all weighing scenarios
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

# # If want to export the results
# file_path = os.path.join(results_path, 'RESULTS_AHP_TOPSIS.xlsx')
# with pd.ExcelWriter(file_path) as writer:
#     bwaise_mcda.perform_scores.to_excel(writer, sheet_name='Score')
#     bwaise_mcda.ranks.to_excel(writer, sheet_name='Rank')


# %%

# =============================================================================
# TOPSIS uncertainties for selected weighing scenarios
# =============================================================================

# TODO: Consider making this a function within MCDA
def run_uncertainty_mcda(mcda, criteria_weights, tech_score_dct):
    scores_df_dct = {}
    ranks_df_dct = {}
    for n, w in criteria_weights.iterrows():
        scores = []
        ranks = []
        for k, v in tech_score_dct.items():
            mcda.tech_scores = v
            mcda.run_MCDA(criteria_weights=w)
            scores.append(mcda.perform_scores)
            ranks.append(mcda.ranks)

        name = w.Ratio.replace(':', '-')
        scores_df_dct[name] = pd.concat(scores).reset_index()
        ranks_df_dct[name] = pd.concat(ranks).reset_index()

    return scores_df_dct, ranks_df_dct

# Note that empty cells (with nan value) are failed simulations
# (i.e., corresponding tech scores are empty)
ratios = ['1:0:0:0:0', '0:1:0:0:0', '0:0:1:0:0', '0:0:0:1:0', '0:0:0:0:1', '1:1:1:1:1']
weights = bwaise_mcda.criteria_weights[bwaise_mcda.criteria_weights.Ratio.isin(ratios)]

scores_df_dct, ranks_df_dct = run_uncertainty_mcda(bwaise_mcda, weights, tech_score_dct)

# # If want to export the results
# file_path = os.path.join(results_path, 'uncertainty/AHP_TOPSIS.xlsx')
# with pd.ExcelWriter(file_path) as writer:
#     for k, v in scores_df_dct.items():
#         v.to_excel(writer, sheet_name=k+' score')
#         ranks_df_dct[k].to_excel(writer, sheet_name=k+' rank')


# %%

# =============================================================================
# Kolmogorov–Smirnov test for TOPSIS uncertainties
# =============================================================================

# TODO: Consider making this a function within MCDA
def run_correlation_test(input_x, input_y, kind,
                         nan_policy='omit', file_path='', **kwargs):
    '''
    Get correlation coefficients between two inputs using `scipy`.

    Parameters
    ----------
    input_x : :class:`pandas.DataFrame`
        The first set of input (typically uncertainty parameters).
    input_y : :class:`pandas.DataFrame`
        The second set of input (typicall scores or ranks).
    kind : str
        The type of test to run, can be "Spearman" for Spearman's rho,
        "Pearson" for Pearson's r, "Kendall" for Kendall's tau,
        or "KS" for Kolmogorov–Smirnov's D.
    nan_policy : str
        - "propagate": returns nan.
        - "raise": raise an error.
        - "omit": drop the pair from analysis.

        .. note::
            This will be ignored for when `kind` is "Pearson" or "Kendall"
            (not supported by `scipy`).

    file_path : str
        If provided, the results will be saved as an Excel file to the given path.
    kwargs : dict
        Other keyword arguments that will be passed to ``scipy``.

    Returns
    -------
    Two :class:`pandas.DataFrame` containing the test statistics and p-values.

    See Also
    --------
    :func:`scipy.stats.spearmanr`

    :func:`scipy.stats.pearsonr`

    :func:`scipy.stats.kendalltau`

    :func:`scipy.stats.kstest`

    '''

    name = kind.lower()
    if name == 'spearman':
        correlation = stats.spearmanr
        col_name = 'rho'
        kwargs['nan_policy'] = nan_policy
    elif name == 'pearson':
        correlation = stats.pearsonr
        col_name = 'r'
    elif name == 'kendall':
        correlation = stats.kendalltau
        col_name = 'tau'
        kwargs['nan_policy'] = nan_policy
    elif name == 'ks':
        correlation = stats.kstest
        col_name = 'd'
    else:
        raise ValueError('kind can only be "Spearman", "Pearson", '
                        f'"Kendall", or "KS", not "{kind}".')

    labels = np.array(
        [[[str(col_x), col_y] for col_x in input_x.columns] for col_y in input_y.columns]
        )[0]

    data = np.array(
        [[correlation(input_x[col_x], input_y[col_y], **kwargs) \
          for col_x in input_x.columns] for col_y in input_y.columns]
        )[0]

    df = pd.DataFrame({
        'Input x': labels[:, 0],
        'Input y': labels[:, 1],
        col_name: data[:, 0],
        'p-value': data[:, 1],
        })

    if file_path:
        df.to_csv(file_path, sep='\t')
    return df


def run_uncertainty_corr(df_dct, kind):
    corr_dct = {}
    for k, v in df_dct.items(): # each key is a weighing scenario
        for i in ('A', 'B', 'C'):
            corr_dct[f'{k} {i}'] = run_correlation_test(
                input_x=param_dct[f'sys{i}'],
                input_y=v[f'Alternative {i}'].to_frame(),
                kind=kind)
    return corr_dct


# Correlation between parameter values and scores
score_corr_dct = run_uncertainty_corr(scores_df_dct, 'KS')

# If want to export the results
file_path = os.path.join(results_path, 'sensitivity/AHP_TOPSIS_KS_scores.xlsx')
with pd.ExcelWriter(file_path) as writer:
    for k, v in score_corr_dct.items():
        v.to_excel(writer, sheet_name=k)


# Correlation between parameter values and ranks
rank_corr_dct = run_uncertainty_corr(ranks_df_dct, 'KS')

# If want to export the results
file_path = os.path.join(results_path, 'sensitivity/AHP_TOPSIS_KS_ranks.xlsx')
with pd.ExcelWriter(file_path) as writer:
    for k, v in rank_corr_dct.items():
        v.to_excel(writer, sheet_name=k)