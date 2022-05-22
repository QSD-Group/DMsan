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
from scipy import stats
from matplotlib import pyplot as plt
from qsdsan.utils import time_printer, save_pickle, colors
from dmsan import AHP, MCDA, path

#!!! NEED TO USE THE SAME SAMPLES ACROSS SYSTEMS

# %%

# =============================================================================
# Utils
# =============================================================================

rng = np.random.default_rng(3221) # set random number generator for reproducible results
criterion_num = 5 # number of criteria
wt_scenario_num = 100 # number of criterion weights considered


def generate_weights(criterion_num, wt_scenario_num, savefig=True):
    # Use randomly generated criteria weights
    wt_sampler1 = stats.qmc.LatinHypercube(d=1, seed=rng)
    n = int(wt_scenario_num/criterion_num)
    wt1 = wt_sampler1.random(n=n) # draw from 0 to 1 for one criterion

    wt_sampler4 = stats.qmc.LatinHypercube(d=(criterion_num-1), seed=rng)
    wt4 = wt_sampler4.random(n=n) # normalize the rest four based on the first criterion
    tot = wt4.sum(axis=1) / ((np.ones_like(wt1)-wt1).transpose())
    wt4 = wt4.transpose()/np.tile(tot, (wt4.shape[1], 1))

    combined = np.concatenate((wt1.transpose(), wt4)).transpose()

    wts = [combined]
    for num in range(criterion_num-1):
        combined = np.roll(combined, 1, axis=1)
        wts.append(combined)

    weights = np.concatenate(wts).transpose()

    # Plot all of the criterion weight scenarios
    if savefig:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(weights, color=colors.Guest.gray.RGBn, linewidth=0.5, alpha=0.5)
        ax.set(title='Criterion Weight Scenarios',
               xlim=(0, 4), ylim=(0, 1), ylabel='Criterion Weights',
               xticks=(0, 1, 2, 3, 4),
               xticklabels=('T', 'RR', 'Env', 'Econ', 'S'))
        from dmsan.gates import figures_path
        fig.savefig(os.path.join(figures_path, f'criterion_weights_{wt_scenario_num}.png'),
                    dpi=300)

    weight_df = pd.DataFrame(weights.transpose(), columns=['T', 'RR', 'Env', 'Econ', 'S'])
    colon = np.full(weight_df.shape[0], fill_value=':', dtype='str')
    comma = np.full(weight_df.shape[0], fill_value=', ', dtype='U2')
    weight_df['Ratio'] = weight_df['Description'] = ''
    for i in ['T', 'RR', 'Env', 'Econ', 'S']:
        ratio = weight_df[i].round(2).astype('str')
        criteria = comma.astype('U4')
        criteria.fill(i)

        if i != 'S':
            weight_df['Ratio'] += ratio + colon
            weight_df['Description'] += ratio + criteria + comma
        else:
            weight_df['Ratio'] += ratio
            weight_df['Description'] += ratio + criteria

    return weight_df


# %%

# =============================================================================
# Baseline
# =============================================================================

# modules = ('biogenic_refinery', 'newgen', 'reclaimer') #!!! need to add in biogenic_refinery when it's done
modules = ('newgen', 'reclaimer')
get_alt_names = lambda module: [f'{module}A', f'{module}B'] if module!='reclaimer' \
    else [f'{module}B', f'{module}C']


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

def get_uncertainty_data(country):
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
        baseline_scores = get_baseline_indicator_scores(country)

        AB = 'AB' if module != 'reclaimer' else 'BC'
        sysA = pd.read_excel(file, f'sys{AB[0]}-results', index_col=0, header=(0, 1))
        sysB = pd.read_excel(file, f'sys{AB[1]}-results', index_col=0, header=(0, 1))
        sys_val = pd.concat((sys_val, sysA[col_names], sysB[col_names]))

    ind_score_dct = {}
    N = sysA.shape[0]
    n = int(len(sys_val)/N)
    for i in range(N):
        dfs = [sys_val.iloc[n*j+i] for j in range(n)]
        simulated = pd.DataFrame(dfs)
        indicator_scores = baseline_scores.copy()
        indicator_scores[varied_inds] = simulated.values
        ind_score_dct[i] = indicator_scores

    return ind_score_dct


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
# Result exporting
# =============================================================================

#!!! PAUSED


# The Excel files could be vary large, try not to use when have thousands of samples,
# especially for saving the uncertainty analysis results
def export_to_excel(country, indicator_weights=True, mcda=True, criterion_weights=True,
                    uncertainty=True):
    if indicator_weights:
        file_path = os.path.join(results_path, 'indicator_weights.xlsx')
        ahp.norm_weights_df.to_excel(file_path, sheet_name='Local weights')
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


def export_to_pickle(country, indicator_scores=True,
                     ahp=True, mcda=True, uncertainty=True):
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



# %%

# =============================================================================
# MCDA
# =============================================================================

ind_score_dct = get_uncertainty_data('China')

# Names of the alternative systems
alt_names = list(ind_score_dct.values())[0].index.to_list()

def run_analyses(save_excel=False):
    global baseline_indicator_scores
    baseline_indicator_scores = get_baseline_indicator_scores()

    for country in ('China', 'India', 'Senegal', 'South Africa', 'Uganda'):

        # Set the local weight of indicators that all three systems score the same
        # to zero (to prevent diluting the scores)
        eq_ind = baseline_indicator_scores.min() == baseline_indicator_scores.max()
        eq_inds = [(i[:-1], i[-1]) for i in eq_ind[eq_ind == True].index]

        ahp = AHP(location_name=country, num_alt=len(alt_names),
                  na_default=0.00001, random_index={})
        for i in eq_inds:
            # Need subtract in `int(i[1])-1` because of 0-indexing
            ahp.init_weights[i[0]][int(i[1]) - 1] = ahp.na_default

        ahp.get_indicator_weights(True)

        global bwaise_mcda
        bwaise_mcda = MCDA(method='TOPSIS', alt_names=alt_names,
                           indicator_weights=ahp.norm_weights_df,
                           indicator_scores=baseline_indicator_scores)
        bwaise_mcda.run_MCDA()

        global weight_df
        weight_df = generate_weights(criterion_num=criterion_num, wt_scenario_num=wt_scenario_num)

        export_to_excel(indicator_weights=True, mcda=True, criterion_weights=True,
                        uncertainty=False, sensitivity=None)

        # Note that empty cells (with nan value) are failed simulations
        # (i.e., corresponding tech scores are empty)
        global score_df_dct, rank_df_dct, winner_df
        score_df_dct, rank_df_dct, winner_df = \
            bwaise_mcda.run_MCDA_multi_scores(criterion_weights=weight_df,
                                              ind_score_dct=ind_score_dct)

        export_to_pickle(ahp=True, mcda=True, uncertainty=True)


if __name__ == '__main__':
    run_analyses(save_excel=False)
