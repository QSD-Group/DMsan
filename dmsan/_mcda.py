#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors:
    Tori Morgan <vlmorgan@illinois.edu>,
    Hannah Lohman <hlohman94@gmail.com>,
    Stetson Rowles <stetsonsc@gmail.com>,
    Yalin Li <zoe.yalin.li@gmail.com>

This model is developed to assist sanitation system research, development, and
deployment. Users of the model need to manually input where exclamation points
(!) are located in the comments (i.e. location, number of alternatives, etc.,
end-user and/or management preference socres, etc.).
"""

# %%

import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from . import data_path, results_path

# Data files associated with the model
data_path_tech_scores = os.path.join(data_path, 'technology_scores.xlsx')
data_path_weight_scenarios = os.path.join(data_path, 'criteria_weight_scenarios.xlsx')

__all__ = ('MCDA',)


# %%

class MCDA:
    '''
    Class for performing multi-criteria decision analysis.

    Parameters
    ----------
    file_path : str
        Path for the Excel data file containing information related to
        criteria weighing, default path (and file) will be used if not provdided.
    alt_names : Iterable
        Names of the alternative systems under consideration.
    method : str
        MCDA method, either TOPSIS (technique for order of preference by similarity
        to ideal solution) or ELECTRE (ELimination Et Choice Translating REality).
    indicator_weights : :class:`pandas.DataFrame`
        Calculated weights for indicators in the considered criteria.
    tech_scores : :class:`pandas.DataFrame`
        Calculated scores for the alternative systems with regard to each indicator.

    Examples
    --------
    NOT READY YET.

    '''
    def __init__(self,  file_path='', alt_names=(), method='TOPSIS',
                 criteria_weights=None, *,
                 indicator_weights, tech_scores):
        path = file_path if file_path else data_path+'/criteria_weight_scenarios.xlsx'
        file = pd.ExcelFile(path)
        read_excel = lambda name: pd.read_excel(file, name) # name is sheet name

        self.alt_names = alt_names
        self.criteria_weights = criteria_weights if criteria_weights else read_excel('weight_scenarios')
        self.indicator_type = read_excel('indicator_type') #!!! The first row of this Excel is not needed anymores
        self.indicator_weights = indicator_weights
        self.tech_scores = tech_scores
        self.method = method
        self._perform_socres = self._ranks = None


    def run_MCDA(self, criteria_weights=None, save=False, file_path='',
                 method=None):
        '''
        MCDA using the set method.

        Parameters
        ----------
        criteria_weights : :class:`pandas.DataFrame`
            Weights for the different criteria, will be defaulted to all of the
            associated criteria in the `criteria_weights` property if left as None.
        save : bool
            If True, the results will be save as an Excel file.
        file_path : str
            Path for the output Excel file, default path will be used if not provided.
        method : str
            MCDA method, will use value set in the `method` property if not provided.
        '''
        method = self.method if not method else method
        if method.upper() == 'TOPSIS':
            self._run_TOPSIS(criteria_weights, save, file_path)
        elif method.upper() == 'ELECTRE':
            self._run_ELECTRE()
        else:
            raise ValueError('`method` can only be "TOPSIS" OR "ELECTRE", '
                             f'not {method}.')


    def _run_TOPSIS(self, criteria_weights=None, save=False, file_path=''):
        cr_wt = self.criteria_weights if criteria_weights is None else criteria_weights
        ind_type = self.indicator_type.iloc[1, :]
        rev_ind_type = np.ones_like(ind_type) - ind_type
        ind_wt = self.indicator_weights

        tech_scores = self.tech_scores
        tech_scores_a = tech_scores.values
        num_ind = tech_scores.shape[1]
        num_alt = tech_scores.shape[0]
        num_cr = cr_wt.shape[0] if len(cr_wt.shape)==2 else 1 # cr_wt will be a Series if only criterion

        # Step 1: Normalize tech scores (vector normalization)
        denominators = np.array([sum(tech_scores_a[:, i]**2)**0.5
                                 for i in range(num_ind)])
        norm_val = np.divide(tech_scores_a, denominators,
                             out=np.zeros_like(tech_scores_a), # fill 0 when denominator is 0
                             where=denominators!=0)

        # Step 2: Rank systems under criteria weighting scenarios
        criteria = ['T', 'RR', 'Econ', 'Env', 'S']
        num_ind_dct = dict.fromkeys(criteria)
        for k in criteria:
            num_ind_dct[k] = len([i for i in ind_wt.columns if i.startswith(k)])

        # For all criteria weighing scenarios and all indicators
        norm_indicator_weights = np.concatenate(
            [np.tile(cr_wt[i], (num_ind_dct[i], 1)) for i in criteria]
            ).transpose() # the shape is (num_of_weighing_scenarios, num_of_indicators)
        norm_indicator_weights *= np.tile(ind_wt, (num_cr, 1))


        #!!! PAUSED, looks like from here and above used in both TOPSIS and ELECTRE

        # For each weighing scenario considering all alternative systems
        scores = []
        ranks = []
        for i in norm_indicator_weights:
            results = i * norm_val

            # Get ideal best and worst values for each indicator,
            # for indicator types, 0 is non-beneficial (want low value)
            # and 1 is beneficial
            min_a = results.min(axis=0)
            max_a = results.max(axis=0)

            # Best would be the max for beneficial indicators
            # and min for non-beneficial indicators
            best_a = (rev_ind_type.values*min_a) + (ind_type.values*max_a)
            worst_a = (ind_type.values*min_a) + (rev_ind_type.values*max_a)

            # Calculate the Euclidean distance from best and worst
            dif_best = (results-np.tile(best_a, (num_alt, 1))) ** 2
            dif_worst = (results-np.tile(worst_a, (num_alt, 1))) ** 2
            d_best = dif_best.sum(axis=1) ** 0.5
            d_worst = dif_worst.sum(axis=1) ** 0.5

            # Calculate performance score
            score = d_worst / (d_best+d_worst)
            rank = (num_alt+1) - rankdata(score).astype(int)

            scores.append(score)
            ranks.append(rank)

        columns = self.alt_names
        score_df = pd.DataFrame(scores, columns=columns)
        rank_df = pd.DataFrame(ranks, columns=columns)

        pre_df = pd.DataFrame({'Ratio': cr_wt.Ratio, 'Description': cr_wt.Description},
                              index=score_df.index)
        score_df = pd.concat([pre_df, score_df], axis=1).reset_index()
        rank_df = pd.concat([pre_df, rank_df], axis=1).reset_index()

        self._perform_socres = score_df
        self._ranks = rank_df

        if save:
            file_path = os.path.join(results_path, 'RESULTS_AHP_TOPSIS.xlsx') if not file_path else file_path
            with pd.ExcelWriter(file_path) as writer:
                score_df.to_excel(writer, sheet_name='Score')
                rank_df.to_excel(writer, sheet_name='Rank')


    def _run_ELECTRE(self, criteria_weights=None):
        '''NOT READY YET.'''
        raise ValueError('Method not ready yet.')


    @property
    def perform_scores(self):
        '''[:class:`pandas.DataFrame`] Calculated performance scores.'''
        if self._perform_socres is None:
            self.run_MCDA()
        return self._perform_socres

    @property
    def ranks(self):
        '''[:class:`pandas.DataFrame`] Calculated ranks.'''
        if self._ranks is None:
            self.run_MCDA()
        return self._ranks