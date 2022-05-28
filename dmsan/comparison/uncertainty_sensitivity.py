#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors:
    Yalin Li <mailto.yalin.li@gmail.com>
    Hannah Lohman <hlohman94@gmail.com>

Run this module for uncertainty and sensitivity analyses.

Two layers of uncertainties are considered in the final performance score:
    1. Ones due to the input parameters in system simulation, which will change
    the technology scores of some indicators.
    2. Ones due to the global weights of different criteria.

Note that uncertainties in the local weights can also be included if desired,
some of the legacy codes herein added uncertainties to the social criterion,
but doing so with all other uncertainties will add too much complexity and
create difficulties in identifying the drivers of the results, thus not included.
"""

import os, numpy as np, pandas as pd
from qsdsan.utils import save_pickle
from dmsan import AHP, MCDA, path
from dmsan.comparison import results_path, figures_path

# Universal settings
rng = np.random.default_rng(3221) # set random number generator for reproducible results
criterion_num = 5 # number of criteria
wt_scenario_num = 100 # number of criterion weights considered


# %%

# =============================================================================
# Baseline
# =============================================================================

# Alternative systems
# modules = ('biogenic_refinery', 'newgen', 'reclaimer') #!!! need to add in biogenic_refinery when it's done
modules = ('newgen', 'reclaimer')
get_alt_names = lambda module: [f'{module}A', f'{module}B'] if module!='reclaimer' \
    else [f'{module}B', f'{module}C']
alt_names = sum([get_alt_names(module) for module in modules], [])
ahp_kwargs = dict(num_alt=len(alt_names), na_default=0.00001, random_index={})


#!!! Should reconstruct and run the uncertainties here to copy samples
# Baseline
def get_baseline_indicator_scores(country):
    indicator_scores = pd.DataFrame()
    indices = []
    for module in modules:
        scores_path = os.path.join(path, f'{module}/scores')
        indicator_scores_path = os.path.join(scores_path, 'other_indicator_scores.xlsx')
        score_file = pd.ExcelFile(indicator_scores_path)
        read_baseline = lambda name: pd.read_excel(score_file, name).expected

        # Technical
        ind_score_T_All = pd.DataFrame([
            read_baseline('user_interface'),
            read_baseline('treatment_type'),
            read_baseline('system_part_accessibility'),
            read_baseline('design_transport'),
            read_baseline('construction_skills'),
            read_baseline('OM_complexity'),
            read_baseline('pop_flexibility'),
            read_baseline('electricity_flexibility'),
            read_baseline('drought_flexibility')
            ]).transpose()

        ind_score_T_All.columns = [f'T{i+1}' for i in range(ind_score_T_All.shape[1])]

        # Resource Recovery
        # Import simulated results
        baseline = pd.read_csv(os.path.join(scores_path, f'{country}/sys_baseline.tsv'),
                               index_col=(0, 1), sep='\t')

        ind_score_RR_All = pd.DataFrame([
            read_baseline('water_reuse'),
            baseline.loc[('Net recovery', 'N')].values,
            baseline.loc[('Net recovery', 'P')].values,
            baseline.loc[('Net recovery', 'K')].values,
            baseline.loc[('Net recovery', 'energy')].values,
            read_baseline('supply_chain')
            ]).transpose()

        ind_score_RR_All.columns = [f'RR{i+1}' for i in range(ind_score_RR_All.shape[1])]

        # Environmental, "H" for hierarchist
        lca_ind = list(zip(['LCA results']*3, ['H_Ecosystems', 'H_Health', 'H_Resources']))
        ind_score_Env_All = pd.DataFrame([
            baseline[baseline.index==lca_ind[0]].values[0], # ecosystem quality
            baseline[baseline.index==lca_ind[1]].values[0], # human health
            baseline[baseline.index==lca_ind[2]].values[0], # resource depletion
            ]).transpose()

        ind_score_Env_All.columns = [f'Env{i+1}' for i in range(ind_score_Env_All.shape[1])]

        # Economic
        ind_score_Econ_All = pd.DataFrame([
            baseline.loc[('TEA results', 'Net cost')].values
            ]).transpose()

        ind_score_Econ_All.columns = ['Econ1']

        # Social
        ind_score_S_All = pd.DataFrame([
            read_baseline('design_job_creation'),
            read_baseline('design_high_pay_jobs'),
            read_baseline('end_user_disposal'),
            read_baseline('end_user_cleaning'),
            read_baseline('privacy'),
            read_baseline('odor'),
            read_baseline('security'),
            read_baseline('management_disposal'),
            read_baseline('management_cleaning')
            ]).transpose()

        ind_score_S_All.columns = [f'S{i+1}' for i in range(ind_score_S_All.shape[1])]
        df = pd.concat([ind_score_T_All, ind_score_RR_All,
                        ind_score_Env_All, ind_score_Econ_All,
                        ind_score_S_All],
                       axis=1)
        indicator_scores = pd.concat((indicator_scores, df))
        indices += get_alt_names(module)
        indicator_scores.index = indices

    return indicator_scores


# %%

# =============================================================================
# Uncertainty
# =============================================================================

# For uncertainties of those simulated (only some certain parameters are affected)
varied_inds = [*[f'RR{i}' for i in range(2, 6)],  # only include indicators that are simulated
               *[f'Env{i}' for i in range(1, 4)],
               'Econ1']  # net user cost

def get_uncertainty_data(country, baseline_indicator_scores=None):
    col_names = [
        ('N recovery', 'Total N'),
        ('P recovery', 'Total P'),
        ('K recovery', 'Total K'),
        ('COD recovery', 'Gas COD'),  # energy, only gas
        ('LCA results', 'H_Ecosystems [points]'),
        ('LCA results', 'H_Health [points]'),
        ('LCA results', 'H_Resources [points]'),
        ('TEA results', 'Annual net cost [USD/cap/yr]'),
        ]

    sys_val = pd.DataFrame()
    for module in modules:
        scores_path = os.path.join(path, f'{module}/scores')
        file_path = os.path.join(scores_path, f'{country}/sys_uncertainties.xlsx')
        file = pd.ExcelFile(file_path)
        baseline_scores = baseline_indicator_scores or get_baseline_indicator_scores(country)

        AB = 'AB' if module != 'reclaimer' else 'BC'
        sysA = pd.read_excel(file, f'sys{AB[0]}-results', index_col=0, header=(0, 1))
        sysB = pd.read_excel(file, f'sys{AB[1]}-results', index_col=0, header=(0, 1))
        sys_val = pd.concat((sys_val, sysA[col_names], sysB[col_names]))

    #!!! NEED TO USE THE SAME SAMPLES ACROSS SYSTEMS
    # also should add lines to save the parameters
    param_dct, ind_score_dct = {}, {}
    N = sysA.shape[0]
    n = int(len(sys_val)/N)
    for i in range(N):
        dfs = [sys_val.iloc[n*j+i] for j in range(n)]
        simulated = pd.DataFrame(dfs)
        indicator_scores = baseline_scores.copy()
        indicator_scores[varied_inds] = simulated.values
        ind_score_dct[i] = indicator_scores

    return param_dct, ind_score_dct


# %%

# =============================================================================
# Sensitivity
# =============================================================================

#!!! Need to rethink how to do sensitivity analysis
# @time_printer
# def run_uncertainty_corr(df_dct, kind):
#     corr_dct = {}
#     # Alternatives cannot be consolidated as they have different parameters
#     for i in ('A', 'B', 'C'):
#         corr_df = pd.DataFrame(param_dct[f'sys{i}'].columns,
#                           columns=pd.MultiIndex.from_arrays([('',), ('Parameter',)],
#                                                             names=('Weights', 'Stats')))

#         stats, p = None, None # (name of the statistical test, 'p-value')
#         result_dfs = []
#         for k, v in df_dct.items(): # each key is a weighing scenario
#             temp_df = bwaise_mcda.correlation_test(
#                 input_x=param_dct[f'sys{i}'],
#                 input_y=v[f'Alternative {i}'].to_frame(),
#                 kind=kind)
#             if not stats:
#                 stats, p = temp_df.columns[-2:]
#             result_dfs.append(temp_df.iloc[:, -2:])

#         df0 = pd.DataFrame(param_dct[f'sys{i}'].columns,
#                           columns=pd.MultiIndex.from_arrays([('',), ('Parameter',)],
#                                                             names=('Weights', 'Stats')))
#         col1 = pd.MultiIndex.from_product([df_dct.keys(), [stats[1], p[1]]],
#                                           names=['Weights', 'Stats'])
#         df1 = pd.concat(result_dfs, axis=1)
#         df1.columns = col1
#         corr_df = pd.concat([df0, df1], axis=1)

#         corr_dct[i] = corr_df

#     return corr_dct


# %%

# =============================================================================
# MCDA with result exporting
# =============================================================================

def run_analyses(country, weight_df=None):
    print(f'\nRunning analyses for country: {country}.')
    country_folder = os.path.join(results_path, country)
    if not os.path.isdir(country_folder): os.mkdir(country_folder)

    ##### Baseline indicator scores and weights #####
    baseline_indicator_scores = get_baseline_indicator_scores(country)

    # Set the local weight of indicators that all systems score the same
    # to zero (to prevent diluting the scores)
    eq_ind = baseline_indicator_scores.min() == baseline_indicator_scores.max()
    eq_inds = eq_ind[eq_ind == True].index.to_list()

    ahp = AHP(location_name=country, **ahp_kwargs)
    for i in eq_inds: ahp.init_weights[i] = ahp.na_default

    norm_weights_df = ahp.get_indicator_weights(return_results=True)

    # Save normalized indicator weights
    file_path = os.path.join(country_folder, 'ahp.pckl')
    save_pickle(ahp, file_path)
    file_path = os.path.join(country_folder, 'indicator_weights.xlsx')
    norm_weights_df.to_excel(file_path, sheet_name='Local weights')

    ##### Baseline performance scores #####
    mcda = MCDA(method='TOPSIS', alt_names=alt_names,
                indicator_weights=norm_weights_df,
                indicator_scores=baseline_indicator_scores)
    mcda.run_MCDA()

    # Save performance scores and ranks
    file_path = os.path.join(country_folder, 'mcda.pckl')
    save_pickle(mcda, file_path)
    file_path = os.path.join(country_folder, 'performance_baseline.xlsx')
    with pd.ExcelWriter(file_path) as writer:
        mcda.performance_scores.to_excel(writer, sheet_name='Score')
        mcda.ranks.to_excel(writer, sheet_name='Rank')

    ##### Performance score uncertainty #####
    # Note that empty cells (with nan value) are failed simulations
    # (i.e., corresponding tech scores are empty)
    param_dct, ind_score_dct = get_uncertainty_data(country)

    # Save parameters
    #!!! Needed for each country?
    file_path = os.path.join(country_folder, 'parameters.pckl')
    save_pickle(ind_score_dct, file_path)

    # Save indicator scores
    file_path = os.path.join(country_folder, 'indicator_scores.pckl')
    save_pickle(ind_score_dct, file_path)

    weight_df = weight_df or mcda.generate_criterion_weights(wt_scenario_num)
    score_df_dct, rank_df_dct, winner_df = mcda.run_MCDA_multi_scores(
        criterion_weights=weight_df, ind_score_dct=ind_score_dct)

    # Save performance scores and ranks
    obj = (score_df_dct, rank_df_dct, winner_df)
    file_path = os.path.join(country_folder, 'performance_uncertainties.pckl')
    save_pickle(obj, file_path)
    file_path = os.path.join(results_path, 'performance_uncertainties.xlsx')

    score_df = pd.DataFrame()
    rank_df = pd.DataFrame()
    for k, v in score_df_dct.items():
        temp_score = pd.DataFrame()
        temp_rank = pd.DataFrame()
        for module in alt_names:
            temp_score[module] = v[module]
            temp_rank[module] = rank_df_dct[k][module]
        temp_score.columns = temp_rank.columns = pd.MultiIndex.from_product([[k], alt_names])


    with pd.ExcelWriter(file_path) as writer:
        winner_df.to_excel(writer, sheet_name='Winner')
        score_df.to_excel(writer, sheet_name='Score')
        rank_df.to_excel(writer, sheet_name='Rank')

    ##### Performance score sensitivity #####
    #TODO
    return ahp, mcda


def run_all_countries(countries):
    # Global weight scenarios
    weight_df = MCDA.generate_criterion_weights(wt_scenario_num=wt_scenario_num)
    file_path = os.path.join(results_path, f'criterion_weights_{wt_scenario_num}.xlsx')
    weight_df.to_excel(file_path, sheet_name='Criterion weights')
    file_path = os.path.join(figures_path, f'criterion_weights_{wt_scenario_num}.png')
    MCDA.plot_criterion_weight_fig(weight_df, path=file_path)

    ahp_dct, mcda_dct = {}, {}
    for country in countries:
        ahp_dct[country], mcda_dct[country] = run_analyses(country)

    return weight_df, ahp_dct, mcda_dct


if __name__ == '__main__':
    #!!! Missing data fro India, Senegal, and South Africa
    # countries = ('China', 'India', 'Senegal', 'South Africa', 'Uganda')
    countries = ('China', 'Uganda',)
    weight_df, ahp_dct, mcda_dct = run_all_countries(countries)
