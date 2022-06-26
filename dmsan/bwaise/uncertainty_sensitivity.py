#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors:
    Yalin Li <mailto.yalin.li@gmail.com>
    Joy Zhang <joycheung1994@gmail.com>

Run this module for uncertainty and sensitivity analyses.

Two layers of uncertainties are considered in the final performance score:
    1. Ones due to the input parameters in system simulation, which will change
    the technology scores of some indicators.
    2. Ones due to the global weights of different criteria.

Note that uncertainties in the local weights can also be included if desired,
some of the legacy codes herein added uncertainties to the social criterion,
but doing so with all other uncertainties will add too much complexity and
create difficulties in identifying the drivers of the results, thus not included.

Part of this module is based on the QSDsan and BioSTEAM packages:
    - https://github.com/QSD-group/QSDsan
    - https://github.com/BioSTEAMDevelopmentGroup/biosteam
"""

import os, numpy as np, pandas as pd
from qsdsan.utils import time_printer, save_pickle
from dmsan import AHP, MCDA
from dmsan.bwaise import scores_path, results_path, figures_path

# Utils
indicator_scores_path = os.path.join(scores_path, 'other_indicator_scores.xlsx')
score_file = pd.ExcelFile(indicator_scores_path)
read_baseline = lambda name: pd.read_excel(score_file, name).expected

rng = np.random.default_rng(3221) # set random number generator for reproducible results

criterion_num = 5 # number of criteria
wt_scenario_num = 1000 # number of criterion weights considered

# `lca_perspective` can be "I", "H", or "E" for
# individualist, hierarchist, or egalitarian
lca_perspective = 'H'
if not lca_perspective.upper() in ('I', 'H', 'E'):
    raise ValueError('`lca_perspective` can only be "I", "H", or "E", '
                     f'not "{lca_perspective}".')


# %%

# =============================================================================
# Technology scores
# =============================================================================

# Baseline
def get_baseline_indicator_scores(lca_perspective=lca_perspective):
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
    baseline = pd.read_csv(# import simulated results
        os.path.join(scores_path, 'sys_baseline.csv'), index_col=(0, 1))

    ind_score_RR_All = pd.DataFrame([
        read_baseline('water_reuse'),
        baseline.loc[('N recovery', 'Total N')].values,
        baseline.loc[('P recovery', 'Total P')].values,
        baseline.loc[('K recovery', 'Total K')].values,
        baseline.loc[('COD recovery', 'Gas COD')].values,
        read_baseline('supply_chain')
        ]).transpose()

    ind_score_RR_All.columns = [f'RR{i+1}' for i in range(ind_score_RR_All.shape[1])]

    # Environmental
    lca_ind = [ind for ind in baseline.index if ind[1].startswith(f'Net emission {lca_perspective.upper()}_')]
    ind_score_Env_All = pd.DataFrame([
        baseline[baseline.index==lca_ind[0]].values[0], # ecosystem quality
        baseline[baseline.index==lca_ind[1]].values[0], # human health
        baseline[baseline.index==lca_ind[2]].values[0], # resource depletion
        ]).transpose()

    ind_score_Env_All.columns = [f'Env{i+1}' for i in range(ind_score_Env_All.shape[1])]

    # Economic
    ind_score_Econ_All = pd.DataFrame([
        baseline.loc[('TEA results', 'Annual net cost [USD/cap/yr]')].values
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

    indicator_scores = pd.concat([ind_score_T_All, ind_score_RR_All,
                                  ind_score_Env_All, ind_score_Econ_All,
                                  ind_score_S_All], axis=1)

    return indicator_scores


# For uncertainties of those simulated (only some certain parameters are affected)
varied_inds = [*[f'RR{i}' for i in range(2, 6)],
               *[f'Env{i}' for i in range(1, 4)],
               'Econ1'] # user net cost


def get_uncertainty_data(lca_perspective=lca_perspective, baseline_scores=None,):
    if not baseline_scores:
        baseline_scores = get_baseline_indicator_scores()

    file_path = os.path.join(scores_path, 'sys_uncertainties.xlsx')
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
        ('COD recovery', 'Gas COD'), # energy, only gas
        ('LCA results', f'Net emission {lca_perspective.upper()}_EcosystemQuality_Total [points/cap/yr]'),
        ('LCA results', f'Net emission {lca_perspective.upper()}_HumanHealth_Total [points/cap/yr]'),
        ('LCA results', f'Net emission {lca_perspective.upper()}_Resources_Total [points/cap/yr]'),
        ('TEA results', 'Annual net cost [USD/cap/yr]'),
        ]

    sysA_val = sysA[col_names]
    sysB_val = sysB[col_names]
    sysC_val = sysC[col_names]

    ind_score_dct = {}
    N = sysA.shape[0]
    for i in range(N):
        simulated = pd.DataFrame([sysA_val.iloc[i],
                                  sysB_val.iloc[i],
                                  sysC_val.iloc[i]]).reset_index(drop=True)
        indicator_scores = baseline_scores.copy()
        indicator_scores[varied_inds] = simulated

        # design_job_creation including unskilled and skilled (constant) jobs,
        # only Alternative B with the alternative wastewater treatment plant
        # has uncertainties
        indicator_scores['S1'] = 5 + paramB[('TEA', 'Unskilled staff num [-]')][i]
        # end_user_disposal, how many times the toilet needed to be emptied each year
        indicator_scores['S3'] = [*[1/paramA[('Pit latrine-A2', 'Pit latrine emptying period [yr]')][i]]*2,
                                  365/paramC[('UDDT-C2', 'UDDT collection period [d]')][i]]
        # privacy, i.e., num of household per toilet
        indicator_scores['S5'] = paramA[('Pit latrine-A2', 'Toilet density [household/toilet]')][i]
        ind_score_dct[i] = indicator_scores

    return param_dct, ind_score_dct

# Uncertainty
param_dct, ind_score_dct = get_uncertainty_data()
sim_num = len(ind_score_dct) # number of system simulations


# %%

# =============================================================================
# TOPSIS baseline for all weighing scenarios
# =============================================================================

# Names of the alternative systems
alt_names = pd.read_excel(indicator_scores_path, sheet_name='user_interface').system

# Baseline
bwaise_ahp = AHP(location_name='Uganda', num_alt=len(alt_names),
                 na_default=0.00001, random_index={})

# # If want to use initial weights different from the default
# bwaise_ahp2 = AHP(location_name='Uganda', num_alt=len(alt_names),
#                   init_weights={'S1': 1}, # include all the different ones here
#                   na_default=0.00001, random_index={})


# %%

# =============================================================================
# Sensitivity
# =============================================================================

#!!! Getting a lot of warnings, this doesn't look right...
@time_printer
def run_uncertainty_corr(df_dct, kind):
    corr_dct = {}
    # Alternatives cannot be consolidated as they have different parameters
    for i in ('A', 'B', 'C'):
        corr_df = pd.DataFrame(param_dct[f'sys{i}'].columns,
                          columns=pd.MultiIndex.from_arrays([('',), ('Parameter',)],
                                                            names=('Weights', 'Stats')))

        stats, p = None, None # (name of the statistical test, 'p-value')
        result_dfs = []
        for k, v in df_dct.items(): # each key is a weighing scenario
            temp_df = bwaise_mcda.correlation_test(
                input_x=param_dct[f'sys{i}'],
                input_y=v[f'Alternative {i}'].to_frame(),
                kind=kind)
            if not stats:
                stats, p = temp_df.columns[-2:]
            result_dfs.append(temp_df.iloc[:, -2:])

        df0 = pd.DataFrame(param_dct[f'sys{i}'].columns,
                          columns=pd.MultiIndex.from_arrays([('',), ('Parameter',)],
                                                            names=('Weights', 'Stats')))
        col1 = pd.MultiIndex.from_product([df_dct.keys(), [stats[1], p[1]]],
                                          names=['Weights', 'Stats'])
        df1 = pd.concat(result_dfs, axis=1)
        df1.columns = col1
        corr_df = pd.concat([df0, df1], axis=1)

        corr_dct[i] = corr_df

    return corr_dct


# %%

# =============================================================================
# Result exporting
# =============================================================================

# The Excel files could be vary large, try not to use when have thousands of samples,
# especially for saving the uncertainty analysis results
def export_to_excel(indicator_weights=True, mcda=True, criterion_weights=True,
                    uncertainty=True, sensitivity='KS'):
    if indicator_weights:
        file_path = os.path.join(results_path, 'indicator_weights.xlsx')
        bwaise_ahp.norm_weights_df.to_excel(file_path, sheet_name='Local weights')
        print(f'\nIndicator weights exported to "{file_path}".')

    if mcda:
        bwaise_mcda.indicator_scores = baseline_indicator_scores
        file_path = os.path.join(results_path, 'performance_baseline.xlsx')
        with pd.ExcelWriter(file_path) as writer:
            bwaise_mcda.performance_scores.to_excel(writer, sheet_name='Score')
            bwaise_mcda.ranks.to_excel(writer, sheet_name='Rank')
        print(f'\nBaseline performance scores exported to "{file_path}".')

    if criterion_weights:
        file_path = os.path.join(results_path, f'criterion_weights_{wt_scenario_num}.xlsx')
        weight_df.to_excel(file_path, sheet_name='Criterion weights')
        print(f'\nCriterion weights exported to "{file_path}".')

    if uncertainty:
        file_path = os.path.join(results_path, 'uncertainty/performance_uncertainties.xlsx')
        with pd.ExcelWriter(file_path) as writer:
            winner_df.to_excel(writer, sheet_name='Winner')

            #!!! This might not work, need to double-check
            breakpoint()
            Score = writer.book.add_worksheet('Score')
            Rank = writer.book.add_worksheet('Rank')
            writer.sheets['Rank'] = Rank
            writer.sheets['Score'] = Score

            col_num = 0
            for k, v in score_df_dct.items():
                v.to_excel(writer, sheet_name='Score', startcol=col_num)
                rank_df_dct[k].to_excel(writer, sheet_name='Rank', startcol=col_num)
                col_num += v.shape[1]+2
        print(f'\Performance score uncertainties exported to "{file_path}".')

    if sensitivity:
        file_path = os.path.join(results_path, f'sensitivity/performance_{sensitivity}_ranks.xlsx')
        with pd.ExcelWriter(file_path) as writer:
            for k, v in rank_corr_dct.items():
                v.to_excel(writer, sheet_name=k)

        print(f'\n{sensitivity} sensitivity results (ranks) exported to "{file_path}".')

        if sensitivity != 'KS':
            file_path = os.path.join(results_path, f'sensitivity/performance_{sensitivity}_scores.xlsx')
            with pd.ExcelWriter(file_path) as writer:
                for k, v in score_corr_dct.items():
                    v.to_excel(writer, sheet_name=k)

            print(f'\n{sensitivity} sensitivity results (scores) exported to "{file_path}".')


def export_to_pickle(parameters=True, indicator_scores=True,
                     ahp=True, mcda=True,
                     uncertainty=True, sensitivity='KS'):
    if parameters:
        file_path = os.path.join(results_path, 'parameters.pckl')
        save_pickle(param_dct, file_path)
        print(f'\nDict of parameter values exported to "{file_path}".')

    if indicator_scores:
        file_path = os.path.join(results_path, 'indicator_scores.pckl')
        save_pickle(ind_score_dct, file_path)
        print(f'\nDict of indicator scores exported to "{file_path}".')

    if ahp:
        file_path = os.path.join(results_path, 'ahp.pckl')
        save_pickle(bwaise_ahp, file_path)
        print(f'\nAHP object exported to "{file_path}".')

    if mcda:
        bwaise_mcda.indicator_scores = baseline_indicator_scores
        file_path = os.path.join(results_path, 'mcda.pckl')
        save_pickle(bwaise_mcda, file_path)
        print(f'\nMCDA object exported to "{file_path}".')

    if uncertainty:
        obj = (score_df_dct, rank_df_dct, winner_df)
        file_path = os.path.join(results_path, 'uncertainty/performance_uncertainties.pckl')
        save_pickle(obj, file_path)
        print(f'\nPerformance score uncertainties exported to "{file_path}".')

    if sensitivity:
        file_path = os.path.join(results_path, f'sensitivity/performance_{sensitivity}_ranks.pckl')
        save_pickle(rank_corr_dct, file_path)
        print(f'\n{sensitivity} sensitivity results (ranks) exported to "{file_path}".')

        if sensitivity != 'KS':
            file_path = os.path.join(results_path, f'sensitivity/performance_{sensitivity}_scores.pckl')
            save_pickle(score_corr_dct, file_path)
            print(f'\n{sensitivity} sensitivity results (scores) exported to "{file_path}".')


# %%

# =============================================================================
# Run all analyses
# =============================================================================

def run_analyses(save_sensitivity_excel=False):
    global baseline_indicator_scores
    baseline_indicator_scores = get_baseline_indicator_scores()

    # # DO NOT DELETE
    # # Legacy code to set the local weight of indicators
    # # that all three systems score the same to zero
    # eq_ind = baseline_indicator_scores.min()==baseline_indicator_scores.max()
    # eq_inds = eq_ind[eq_ind==True].index
    # for i in eq_inds:
    #     bwaise_ahp.init_weights[i] = bwaise_ahp.na_default
    # bwaise_ahp.get_indicator_weights(return_results=False)

    global bwaise_mcda
    bwaise_mcda = MCDA(method='TOPSIS', alt_names=alt_names,
                       indicator_weights=bwaise_ahp.norm_weights_df,
                       indicator_scores=baseline_indicator_scores)
    bwaise_mcda.run_MCDA()
    # bwaise_mcda.run_MCDA(criterion_weights=[0.15]*4+[0.4]) # for consistency check

    global weight_df
    weight_df = bwaise_mcda.generate_criterion_weights(wt_scenario_num)
    file_path = os.path.join(figures_path, f'criterion_weights_{wt_scenario_num}.png')
    MCDA.plot_criterion_weight_fig(weight_df, path=file_path, color='gray')

    export_to_excel(indicator_weights=True, mcda=True, criterion_weights=True,
                    uncertainty=False, sensitivity=None)

    # Note that empty cells (with nan value) are failed simulations
    # (i.e., corresponding tech scores are empty)
    global score_df_dct, rank_df_dct, winner_df
    score_df_dct, rank_df_dct, winner_df = \
        bwaise_mcda.run_MCDA_multi_scores(criterion_weights=weight_df,
                                          ind_score_dct=ind_score_dct)

    #!!! No longer run Spearman with ranks data
    kind = 'Spearman'
    global score_corr_dct
    score_corr_dct = run_uncertainty_corr(score_df_dct, kind)
    if save_sensitivity_excel: # too large, prefer not to do it
        export_to_excel(indicator_weights=True, mcda=True, uncertainty=False, sensitivity='Spearman')
    export_to_pickle(ahp=True, mcda=True, uncertainty=True, sensitivity='Spearman')

    kind = 'KS'
    global rank_corr_dct
    rank_corr_dct = run_uncertainty_corr(rank_df_dct, kind)
    if save_sensitivity_excel: # too large, prefer not to do it
        export_to_excel(indicator_weights=False, mcda=False, uncertainty=False, sensitivity='KS')

    export_to_pickle(ahp=False, mcda=False, uncertainty=False, sensitivity='KS')


if __name__ == '__main__':
    run_analyses(save_sensitivity_excel=False)