#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 17:22:03 2021

@authors:
    Yalin Li <zoe.yalin.li@gmail.com>
    Joy Cheung <joycheung1994@gmail.com>

Part of this module is based on the BioSTEAM and QSD packages:
https://github.com/BioSTEAMDevelopmentGroup/biosteam
https://github.com/QSD-group/QSDsan
"""

import os, pickle
import numpy as np
import pandas as pd
from scipy import stats
from qsdsan.utils import time_printer
from dmsan import AHP, MCDA
from dmsan.bwaise import scores_path, results_path


# Util function
tech_scores_path = os.path.join(scores_path, 'other_tech_scores.xlsx')
score_file = pd.ExcelFile(tech_scores_path)
read_baseline = lambda name: pd.read_excel(score_file, name).expected
rng = np.random.default_rng(3221) # set random number generator for reproducible results
criteria_num = 5 # number of criteria
mcda_num = 10 # number of criteria weights considered


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

    tech_score_T_All.columns = [f'T{i+1}' for i in range(tech_score_T_All.shape[1])]

    # Resource Recovery
    # Import simulated results
    baseline = pd.read_csv(os.path.join(scores_path, 'sys_baseline.tsv'),
                           index_col=(0, 1), sep='\t')

    tech_score_RR_All = pd.DataFrame([
        read_baseline('water_reuse'),
        baseline.loc[('Net recovery', 'N')].values,
        baseline.loc[('Net recovery', 'P')].values,
        baseline.loc[('Net recovery', 'K')].values,
        baseline.loc[('Net recovery', 'energy')].values,
        read_baseline('supply_chain')
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

    tech_score_S_All.columns = [f'S{i+1}' for i in range(tech_score_S_All.shape[1])]

    # `tech_scores` is `Tech_Scores_compiled` in the original script,
    # values checked to be the same as the original script
    tech_scores = pd.concat([tech_score_T_All, tech_score_RR_All, tech_score_Econ_All,
                             tech_score_Env_All, tech_score_S_All], axis=1)

    return tech_scores

baseline_tech_scores = get_baseline_tech_scores()


# For uncertainties of those simulated (only some certain parameters are affected)
varied_inds = [*[f'RR{i}' for i in range(2, 6)],
               *[f'Env{i}' for i in range(1, 4)],
               'Econ1'] # user net cost
# varied_idx = baseline_tech_scores.columns.get_indexer_for(varied_inds)

def get_uncertainty_data(lca_perspective='H', baseline_scores=None):
    check_lca(lca_perspective)

    if not baseline_scores:
        baseline_scores = get_baseline_tech_scores()

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
        # design_job_creation including unskilled and skilled (constant) jobs,
        # only Alternative B with the alternative wastewater treatment plant
        # has uncertainties
        tech_scores['S1'] = 5 + paramB[('TEA', 'Unskilled staff num [-]')][i]
        # end_user_disposal, how many times the toilet needed to be emptied each year
        tech_scores['S3'] = [*[paramA[('Pit latrine-A2', 'Pit latrine emptying period [years]')][i]]*2,
                             365/paramC[('UDDT-C2', 'UDDT collection period [days]')][i]]
        # privacy, i.e., num of household per toilet
        tech_scores['S5'] = paramA[('Excretion-A1', 'Toilet density [household/toilet]')][i]
        tech_score_dct[i] = tech_scores

    return param_dct, tech_score_dct

param_dct, tech_score_dct = get_uncertainty_data()
sim_num = len(tech_score_dct) # number of system simulations


# %%

# =============================================================================
# TOPSIS baseline for all weighing scenarios
# =============================================================================

# Names of the alternative systems
alt_names = pd.read_excel(tech_scores_path, sheet_name='user_interface').system

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


# %%

# =============================================================================
# TOPSIS uncertainties for selected weighing scenarios
# =============================================================================

# Generate the same number of local weights as in system simulation,
# three of the local weights in the social criterion hve uncertainties
@time_printer
def get_AHP_weights(N):
    # Baseline for S4-S6 (cleaning preference, privacy, odor and flies)
    b = np.array(bwaise_ahp.init_weights['S'][3:6])
    b = np.tile(b, (N, 1))
    lower = b * 0.75 # 75% of b as the lower bound
    diff = b * 0.5 # 125% of b as the upper bound minus the lower bound
    ahp_sampler = stats.qmc.LatinHypercube(d=3, seed=rng)
    ahp_sample = ahp_sampler.random(n=N)

    S_val_samples = lower + diff*ahp_sample

    AHP_dct = {}
    for i in range(N):
        bwaise_ahp.init_weights['S'][3:6] = S_val_samples[i]
        AHP_dct[i] = bwaise_ahp.get_AHP_weights(True)

    return AHP_dct

AHP_dct = get_AHP_weights(N=sim_num)


# TODO: This should be made into a function within `MCDA`
# TODO: remove extra index column
@time_printer
def run_uncertainty_mcda(mcda, criteria_weights=None, tech_score_dct={}, print_time=True):
    if criteria_weights is None:
        criteria_weights = mcda.criteria_weights

    score_df_dct, rank_df_dct, winner_df_dct = {}, {}, {}
    for n, w in criteria_weights.iterrows():
        scores, ranks, winners = [], [], []

        for k, v in tech_score_dct.items():
            # Update local weights calculated from AHP
            bwaise_ahp._norm_weights_df = AHP_dct[k]

            mcda.tech_scores = v
            mcda.run_MCDA(criteria_weights=w)
            scores.append(mcda.perform_scores)
            ranks.append(mcda.ranks)
            winners.append(mcda.winners.Winner.values.item())

        name = w.Ratio
        score_df_dct[name] = pd.concat(scores).reset_index(drop=True)
        rank_df_dct[name] = pd.concat(ranks).reset_index(drop=True)
        winner_df_dct[name] = winners

    winner_df = pd.DataFrame.from_dict(winner_df_dct)

    return score_df_dct, rank_df_dct, winner_df


# Use randomly generated criteria weights
wt_sampler1 = stats.qmc.LatinHypercube(d=1, seed=rng)
n = int(mcda_num/criteria_num)
wt1 = wt_sampler1.random(n=n) # draw from 0 to 1 for one criterion

wt_sampler4 = stats.qmc.LatinHypercube(d=(criteria_num-1), seed=rng)
wt4 = wt_sampler4.random(n=n) # normalize the rest four based on the first criterion
tot = wt4.sum(axis=1) / ((np.ones_like(wt1)-wt1).transpose())
wt4 = wt4.transpose()/np.tile(tot, (wt4.shape[1], 1))

combined = np.concatenate((wt1.transpose(), wt4)).transpose()

wts = [combined]
for num in range(criteria_num-1):
    combined = np.roll(combined, 1, axis=1)
    wts.append(combined)

weights = np.concatenate(wts).transpose()

# fig0, ax0 = plt.subplots(figsize=(8, 4.5))
# ax0.plot(weights, linewidth=0.5)

# # Using chaospy
# wt_sampler = stats.qmc.LatinHypercube(d=criteria_num, seed=rng)
# weights = wt_sampler.random(n=mcda_num)
# fig1, ax1 = plt.subplots(figsize=(8, 4.5))
# ax1.plot(weights.transpose(), linewidth=0.5)

# weights = weights.transpose()/np.tile(weights.sum(axis=1), (weights.shape[1], 1))
# fig2, ax2 = plt.subplots(figsize=(8, 4.5))
# ax2.plot(weights, linewidth=0.5)

# # Using chaospy
# import chaospy
# weights = chaospy.create_latin_hypercube_samples(order=1000, dim=criteria_num)
# fig3, ax3 = plt.subplots(figsize=(8, 4.5))
# ax3.plot(weights, linewidth=0.5)

# weights = weights/np.tile(weights.transpose().sum(axis=1).transpose(), (weights.shape[0], 1))
# fig4, ax4 = plt.subplots(figsize=(8, 4.5))
# ax4.plot(weights, linewidth=0.5)

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


# Note that empty cells (with nan value) are failed simulations
# (i.e., corresponding tech scores are empty)
score_df_dct, rank_df_dct, winner_df = \
    run_uncertainty_mcda(mcda=bwaise_mcda,
                         criteria_weights=weight_df,
                         tech_score_dct=tech_score_dct)


# %%

# =============================================================================
# Kolmogorov–Smirnov test for TOPSIS uncertainties
# =============================================================================

# TODO: This should be made into a function within `MCDA`
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

        .. note::
            If running KS test, then input_y should be the ranks (i.e., not scores),
            the x inputs will be divided into two groups - ones that results in
            a given alternative to be ranked first vs. the rest.

    nan_policy : str
        - "propagate": returns nan.
        - "raise": raise an error.
        - "omit": drop the pair from analysis.

        .. note::
            This will be ignored for when `kind` is "Pearson" or "Kendall"
            (not supported by `scipy`).

    file_path : str
        If provided, the results will be saved as a csv file to the given path.
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
    df = pd.DataFrame(input_x.columns,
                      columns=pd.MultiIndex.from_arrays([('',), ('Parameter',)],
                                                        names=('Y', 'Stats')))

    name = kind.lower()
    if name == 'spearman':
        func = stats.spearmanr
        stats_name = 'rho'
        kwargs['nan_policy'] = nan_policy
    elif name == 'pearson':
        func = stats.pearsonr
        stats_name = 'r'
    elif name == 'kendall':
        func = stats.kendalltau
        stats_name = 'tau'
        kwargs['nan_policy'] = nan_policy
    elif name == 'ks':
        if not int(input_y.iloc[0, 0])==input_y.iloc[0, 0]:
            raise ValueError('For KS test, `input_y` should be the ranks, not scores.')

        func = stats.kstest
        stats_name = 'D'

        alternative = kwargs.get('alternative') or 'two_sided'
        mode = kwargs.get('mode') or 'auto'
    else:
        raise ValueError('kind can only be "Spearman", "Pearson", '
                        f'or "Kendall", not "{kind}".')

    for col_y in input_y.columns:
        if name == 'ks':
            y = input_y[col_y]
            i_win, i_lose = input_x.loc[y==1], input_x.loc[y!=1]

            if len(i_win) == 0 or len(i_lose) == 0:
                df[(col_y, stats_name)] = df[(col_y, 'p-value')] = None
                continue

            else:
                data = np.array([func(i_win.loc[:, col_x], i_lose.loc[:, col_x],
                                  alternative=alternative, mode=mode, **kwargs) \
                                 for col_x in input_x.columns])
        else:
            data = np.array([func(input_x[col_x], input_y[col_y], **kwargs)
                             for col_x in input_x.columns])

        df[(col_y, stats_name)] = data[:, 0]
        df[(col_y, 'p-value')] = data[:, 1]

    if file_path:
        df.to_csv(file_path, sep='\t')
    return df


@time_printer
def run_uncertainty_corr(df_dct, kind, print_time=True):
    corr_dct = {}
    # Alternatives cannot be consolidated as they have different parameters
    for i in ('A', 'B', 'C'):
        corr_df = pd.DataFrame(param_dct[f'sys{i}'].columns,
                          columns=pd.MultiIndex.from_arrays([('',), ('Parameter',)],
                                                            names=('Weights', 'Stats')))

        for k, v in df_dct.items(): # each key is a weighing scenario
            temp_df = run_correlation_test(
                input_x=param_dct[f'sys{i}'],
                input_y=v[f'Alternative {i}'].to_frame(),
                kind=kind)
            stats, p = temp_df.columns[-2:]
            corr_df[(k, stats[1])] = temp_df[stats]
            corr_df[(k, p[1])] = temp_df[p]

        corr_dct[i] = corr_df

    return corr_dct


kind = 'Spearman'
score_corr_dct = run_uncertainty_corr(score_df_dct, kind)
rank_corr_dct = run_uncertainty_corr(rank_df_dct, kind)

kind = 'KS'
rank_corr_dct = run_uncertainty_corr(rank_df_dct, kind)


# %%

# =============================================================================
# Result exporting
# =============================================================================

# The Excel files could be vary large, try not to use when have thousands of samples,
# especially for saving the uncertainty analysis results
def export_to_excel(baseline=True, uncertainty=True, sensitivity='KS'):
    if baseline:
        file_path = os.path.join(results_path, 'AHP_TOPSIS_baseline.xlsx')
        with pd.ExcelWriter(file_path) as writer:
            bwaise_mcda.perform_scores.to_excel(writer, sheet_name='Score')
            bwaise_mcda.ranks.to_excel(writer, sheet_name='Rank')
        print(f'\nBaseline MCDA results exported to "{file_path}".')

    if uncertainty:
        file_path = os.path.join(results_path, 'uncertainty/AHP_TOPSIS.xlsx')
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
        print(f'\nUncertainty MCDA results exported to "{file_path}".')

    if sensitivity:
        file_path = os.path.join(results_path, f'sensitivity/AHP_TOPSIS_{sensitivity}_ranks.xlsx')
        with pd.ExcelWriter(file_path) as writer:
            for k, v in rank_corr_dct.items():
                v.to_excel(writer, sheet_name=k)

        print(f'\n{sensitivity} sensitivity results (ranks) exported to "{file_path}".')

        if sensitivity != 'KS':
            file_path = os.path.join(results_path, f'sensitivity/AHP_TOPSIS_{sensitivity}_scores.xlsx')
            with pd.ExcelWriter(file_path) as writer:
                for k, v in score_corr_dct.items():
                    v.to_excel(writer, sheet_name=k)

            print(f'\n{sensitivity} sensitivity results (scores) exported to "{file_path}".')

export_to_excel(baseline=True, uncertainty=False, sensitivity='Spearman')
export_to_excel(baseline=False, uncertainty=False, sensitivity='KS')


# Note that Python pickle files may be version-specific,
# (e.g., if saved using Python 3.7, cannot open on Python 3.8)
# and cannot be opened outside of Python,
# but takes much less time to load/save than Excel files
# https://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk
def export_to_pickle(baseline=True, uncertainty=True, sensitivity='KS'):
    def save(obj, path):
        f = open(path, 'wb')
        pickle.dump(obj, f)
        f.close()

    if baseline:
        file_path = os.path.join(results_path, 'bwaise_mcda.pckl')
        save(bwaise_mcda, file_path)
        print(f'\nBaseline MCDA results exported to "{file_path}".')

    if uncertainty:
        obj = (score_df_dct, rank_df_dct, winner_df)
        file_path = os.path.join(results_path, 'uncertainty/AHP_TOPSIS.pckl')
        save(obj, file_path)
        print(f'\nUncertainty MCDA results exported to "{file_path}".')

    if sensitivity:
        file_path = os.path.join(results_path, f'sensitivity/AHP_TOPSIS_{sensitivity}_ranks.pckl')
        save(rank_corr_dct, file_path)
        print(f'\n{sensitivity} sensitivity results (ranks) exported to "{file_path}".')

        if sensitivity != 'KS':
            file_path = os.path.join(results_path, f'sensitivity/AHP_TOPSIS_{sensitivity}_scores.pckl')
            save(score_corr_dct, file_path)
            print(f'\n{sensitivity} sensitivity results (scores) exported to "{file_path}".')

export_to_pickle(baseline=True, uncertainty=True, sensitivity='Spearman')
export_to_pickle(baseline=False, uncertainty=False, sensitivity='KS')