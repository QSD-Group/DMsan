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

Run this module for uncertainty and sensitivity analyses.

Two layers of uncertainties are considered in the final performance score:
    1. Ones due to the input parameters in system simulation, which will change
    the technology scores of some indicators.
    2. Ones due to the global weights of different criteria.

Note that uncertainties in the local weights can also be included if desired,
some of the legacy codes herein added uncertainties to the social criterion,
but doing so with all other uncertainties will add too much complexity and
create difficulties in identifying the drivers of the results, thus not included.
'''

import os, numpy as np, pandas as pd
from collections import OrderedDict
from qsdsan.utils import save_pickle, load_pickle, time_printer
from dmsan import AHP, MCDA
from dmsan.comparison import scores_path, results_path, figures_path

# Universal settings
rng = np.random.default_rng(3221)  # set random number generator for reproducible results
criterion_num = 5  # number of criteria
wt_scenario_num = 100  # number of criterion weights considered

mcda_countries = ('China', 'India', 'Senegal', 'South Africa', 'Uganda')


# %%

# =============================================================================
# Baseline
# =============================================================================

col_names = {
    'N': ('N recovery', 'Total N [% N]'),
    'P': ('P recovery', 'Total P [% P]'),
    'K': ('K recovery', 'Total K [% K]'),
    # 'Energy': (), # no system can recovery energy for external usage
    'Ecosystems': ('LCA results', 'H_Ecosystems [points/cap/yr]'),
    'Health': ('LCA results', 'H_Health [points/cap/yr]'),
    'Resources': ('LCA results', 'H_Resources [points/cap/yr]'),
    'Cost': ('TEA results', 'Annual net cost [USD/cap/yr]'),
    }
num_simulated_ind = len(col_names)

alt_col_names = [
    *[f'RR{i}' for i in range(2, 5)],
    *[f'Env{i}' for i in range(1, 4)],
    'Econ1',
    ]
col_names.update({alt_col_names[n]:val for n, val in enumerate(col_names.values())})
get_simulated_data = lambda df, indicator: df.loc[col_names[indicator]].values


def split_df_by_country(df):
    '''Divide a dataframe into a dict based on the countries.'''
    cols = df.columns
    global countries
    countries = sorted(set((col.split('_')[-1] for col in cols)))
    countries = ['general', *(i for i in countries if i!='general')]
    dct = OrderedDict.fromkeys(countries)
    for country in countries:
        filtered = df.filter(regex=f'_{country}').copy()
        filtered.columns = [col.split('_')[0] for col in filtered.columns]
        dct[country] = filtered
    return dct


def get_baseline_indicator_scores():
    # Simulated indicator scores
    simulated_path = os.path.join(scores_path, 'simulated_baseline.csv')
    simulated = pd.read_csv(simulated_path, index_col=(0, 1))
    simulated_dct = split_df_by_country(simulated)

    # Assigned indicator scores
    assigned_path = os.path.join(scores_path, 'other_indicator_scores.xlsx')
    assigned_file = pd.ExcelFile(assigned_path)
    read_baseline = lambda name: pd.read_excel(assigned_file, name).expected

    baseline_dct = OrderedDict() # to save the compiled results
    for country, df in simulated_dct.items():
        # Technical
        T = pd.DataFrame([
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
        T.columns = [f'T{i+1}' for i in range(T.shape[1])]

        # Resource recovery
        RR = pd.DataFrame([
            read_baseline('water_reuse'),
            get_simulated_data(df, 'N'),
            get_simulated_data(df, 'P'),
            get_simulated_data(df, 'K'),
            np.zeros(T.shape[0]),
            # get_simulated_data(df, 'Energy'),
            read_baseline('supply_chain')
            ]).transpose()
        RR.columns = [f'RR{i+1}' for i in range(RR.shape[1])]

        # Environmental, "H" for hierarchist
        Env = pd.DataFrame([
            get_simulated_data(df, 'Ecosystems'),
            get_simulated_data(df, 'Health'),
            get_simulated_data(df, 'Resources'),
            ]).transpose()
        Env.columns = [f'Env{i+1}' for i in range(Env.shape[1])]

        # Economic
        Econ = pd.DataFrame([get_simulated_data(df, 'Cost')]).transpose()
        Econ.columns = ['Econ1']

        # Social
        S = pd.DataFrame([
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
        S.columns = [f'S{i+1}' for i in range(S.shape[1])]

        # All
        compiled = pd.concat([T, RR, Env, Econ, S], axis=1)
        compiled.index = df.columns
        baseline_dct[country] = compiled

    global alt_names, num_alt
    alt_names = list(df.columns)
    num_alt = len(alt_names)

    return baseline_dct


# %%

# =============================================================================
# MCDA with result exporting
# =============================================================================

@time_printer
def run_analyses(weight_df=None):
    # Used to cache all data
    data = {}

    # Baseline
    baseline_dct = get_baseline_indicator_scores()
    countries = list(baseline_dct.keys())

    # Uncertainties
    #!!! This might take a long time if the Excel is big, consider better alternatives
    uncertainty_path = os.path.join(scores_path, 'simulated_uncertainties.xlsx')
    uncertainty_file = pd.ExcelFile(uncertainty_path)

    def compile_uncertainty_data(country, kind):
        data = [pd.read_excel(uncertainty_file,
                              f'{alt}_{country}_{kind}',
                              index_col=0, header=(0,1),)
                for alt in alt_names]
        for df in data:
            df.drop(columns=[('LCA results', 'GlobalWarming [kg CO2-eq/cap/yr]')], inplace=True)
        compiled = pd.concat(data, axis=1, keys=alt_names)
        return compiled

    result_shape = (num_alt, num_simulated_ind)

    # Global weight scenarios
    weight_df = weight_df or MCDA.generate_criterion_weights(wt_scenario_num)

    weight_path = os.path.join(results_path, f'criterion_weights_{wt_scenario_num}.xlsx')
    weight_df.to_excel(weight_path, sheet_name='Criterion weights')
    file_path = os.path.join(figures_path, f'criterion_weights_{wt_scenario_num}.png')
    MCDA.plot_criterion_weight_fig(weight_df, path=file_path, color='k')

    # [ratio1, ratio2, ..., ratioN] -> ratio1:ratio2:...ratioN
    cr_weights = weight_df.values.tolist()
    cr_weights =[':'.join(str(weight).strip('[]').split(', ')) for weight in cr_weights]

    for country in mcda_countries:
        if country not in countries:
            raise ValueError(f'No simulated scores for country "{country}", '
                             'please run simulation.')
        else: countries.remove(country)

        print(f'\nRunning analyses for country: {country}.')
        mcda_baseline, mcda_uncertainty = {}, {}

        ##### Baseline indicator scores and weights #####
        baseline_scores = baseline_dct[country]
        ahp = AHP(location_name=country, num_alt=num_alt)
        mcda_baseline ['AHP'] = ahp
        ind_weights_df = ahp.get_indicator_weights(return_results=True)
        mcda_baseline['indicator weights'] = ind_weights_df

        # # DO NOT DELETE
        # # Set the local weight of indicators that all systems score the same
        # # to zero (to prevent diluting the scores)
        # eq_ind = baseline_scores.min() == baseline_scores.max()
        # eq_inds = eq_ind[eq_ind==True].index.to_list()
        # for i in eq_inds: ahp.init_weights[i] = ahp.na_default

        ##### Baseline performance scores #####
        mcda = MCDA(
            method='TOPSIS',
            alt_names=alt_names,
            indicator_weights=ind_weights_df,
            indicator_scores=baseline_scores
            )
        mcda.run_MCDA()
        mcda_baseline['mcda'] = mcda
        mcda_baseline['performance scores'] = mcda.performance_scores
        mcda_baseline['performance ranks'] = mcda.ranks
        mcda_baseline['winners'] = mcda.winners

        ##### Performance score uncertainty #####
        # Note that empty cells (with nan value) are failed simulations
        # (i.e., corresponding indicator scores are empty)
        uncertainty_ind_scores = compile_uncertainty_data(country, 'results')
        ind_score_dct = {}
        for n in uncertainty_ind_scores.index:
            scores = baseline_scores.copy()
            # Adjust the order (put TEA cost to the last as Econ1)
            df = uncertainty_ind_scores.iloc[n].values.reshape(result_shape)
            reshaped = np.concatenate((df[:, :3], df[:, 4:], df[:, 3].reshape(df.shape[0], 1)), axis=1)
            scores.loc[:, alt_col_names] = reshaped
            ind_score_dct[n] = scores
        mcda_uncertainty['ind_score_dct'] = ind_score_dct

        score_df_dct, rank_df_dct, winner_df = mcda.run_MCDA_multi_scores(
            criterion_weights=weight_df,
            ind_score_dct=ind_score_dct,
            print_time=False,
            )
        mcda_uncertainty['score_df_dct'] = score_df_dct
        mcda_uncertainty['rank_df_dct'] = rank_df_dct
        winner_df.index = uncertainty_ind_scores.index
        mcda_uncertainty['winner_df'] = winner_df

        # # DO NOT DELETE
        # ##### Performance score sensitivity #####
        # uncertainty_params = compile_uncertainty_data(country, 'params')
        # spearman_dct = {}

        # for alt in alt_names:
        #     lst = []
        #     for wt, scores in score_df_dct.items():
        #         lst.append(
        #             mcda.correlation_test(
        #                 input_x=uncertainty_params.filter(regex=alt),
        #                 input_y=getattr(scores, alt),
        #                 kind='Spearman',
        #             ))
        #     params = lst[0].loc[:, ('', 'Parameter')]
        #     df0 = params.to_frame(name='Parameter')
        #     rhos = [df.loc[:, (0, 'rho')] for df in lst]
        #     df1 = pd.concat(rhos, axis=1)
        #     df1.columns = cr_weights
        #     df = pd.concat(([df0, df1]), axis=1)
        #     spearman_dct[alt] = df
        # mcda_uncertainty['spearman_dct'] = spearman_dct

        ##### Cache results #####
        data[country] = {'baseline': mcda_baseline, 'uncertainty': mcda_uncertainty}

    baseline_performance_path = os.path.join(results_path, 'baseline_performance.xlsx')
    with pd.ExcelWriter(baseline_performance_path) as writer:
        ind_weights = []

        for n, country in enumerate(countries):
            mcda_baseline = data[country]['baseline']
            if n == 0:
                winner_df = pd.concat([data[country]['baseline']['winners']
                                       for country in countries], axis=1)
                winner_df.columns = countries
                winner_df.to_excel(writer, sheet_name='Winner')
            ind_weights.append(mcda_baseline['indicator weights'])
            mcda_baseline['performance scores'].to_excel(writer, sheet_name=f'{country}_score')
            mcda_baseline['performance ranks'].to_excel(writer, sheet_name=f'{country}_rank')

        compiled_df = pd.concat(ind_weights)
        compiled_df.index = countries
        compiled_df.to_csv(os.path.join(results_path, 'baseline_indicator_weights.csv'))

    uncertainty_winner_path = os.path.join(results_path, 'uncertainty_winners.xlsx')
    with pd.ExcelWriter(uncertainty_winner_path) as writer:
        for country in countries:
            data[country]['uncertainty']['winner_df'].to_excel(writer, sheet_name=f'{country}')

    # # DO NOT DELETE
    # spearman_path = os.path.join(results_path, 'spearman.xlsx')
    # with pd.ExcelWriter(spearman_path) as writer:
    #     for country, country_data in data.items():
    #         uncertainty_data = country_data['uncertainty']
    #         spearman_dct = uncertainty_data['spearman_dct']
    #         for alt, df in spearman_dct.items():
    #             df.to_excel(writer, sheet_name=f'{alt}_{country}')

    save_pickle(data, os.path.join(results_path, 'data.pckl'))

    return data


if __name__ == '__main__':
    data = run_analyses()

    # # To load saved data
    # data = load_pickle(os.path.join(results_path, 'data.pckl'))