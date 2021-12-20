#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors:
    Tori Morgan <vlmorgan@illinois.edu>,
    Hannah Lohman <hlohman94@gmail.com>,
    Stetson Rowles <stetsonsc@gmail.com>,
    Yalin Li <zoe.yalin.li@gmail.com>

This module is used to perform calculate performance score and global sensitivity analysis.
"""

# %%

import os
import numpy as np
import pandas as pd
from scipy import stats
from qsdsan.utils import time_printer
from . import data_path, results_path

__all__ = ('MCDA',)


# %%

class MCDA:
    '''
    Class for performing multi-criteria decision analysis.

    Parameters
    ----------
    file_path : str
        Path for the Excel data file containing information related to
        criteria weighing, default path (and file) will be used if not provided.
    alt_names : Iterable
        Names of the alternative systems under consideration.
    method : str
        MCDA method, either TOPSIS (technique for order of preference by similarity
        to ideal solution) or ELECTRE (ELimination Et Choice Translating REality).
    indicator_weights : :class:`pandas.DataFrame`
        Calculated weights for indicators in the considered criteria.
    indicator_type : :class:`pandas.DataFrame`
        Columns should be the code of the indicators as used in :class:`DMsan.,
        values should by either "1" (beneficial) or "0" (non-beneficial).
        For beneficial indicators, the higher the technology score is, the better;
        and vice versa for non-beneficial factors.
    tech_scores : :class:`pandas.DataFrame`
        Calculated scores for the alternative systems with regard to each indicator.
    criterion_weights : :class:`pandas.DataFrame`
        Weight scenarios for the different criteria,
        will use default scenarios if not provided.

    Examples
    --------
    NOT READY YET.

    '''
    def __init__(self,  file_path='', alt_names=(), method='TOPSIS',
                 *, indicator_weights, tech_scores, indicator_type=None,
                 criterion_weights=None):
        path = file_path if file_path else os.path.join(data_path, 'criteria_and_indicators.xlsx')
        file = pd.ExcelFile(path)
        read_excel = lambda name: pd.read_excel(file, name) # name is sheet name

        self.alt_names = alt_names
        self.indicator_weights = indicator_weights
        self._default_definitions = defs = read_excel('definitions')
        self.indicator_type = pd.DataFrame({
                defs.variable[i]: defs.category_binary[i] for i in defs.index
                }, index=[0])
        self.tech_scores = tech_scores
        self.criterion_weights = criterion_weights if criterion_weights else read_excel('weight_scenarios')
        self.method = method
        self._perform_socres = self._ranks = self._winners = None


    def run_MCDA(self, criterion_weights=None, method=None,
                 save=False, file_path=''):
        '''
        MCDA using the set method.

        Parameters
        ----------
        criterion_weights : :class:`pandas.DataFrame`
            Weights for the different criteria, will be defaulted to all of the
            associated criteria in the `criterion_weights` property if left as None.
        method : str
            MCDA method, will use value set in the `method` property if not provided.
        save : bool
            If True, the results will be save as an Excel file.
        file_path : str
            Path for the output Excel file, default path will be used if not provided.
        '''
        method = self.method if not method else method
        if method.upper() == 'TOPSIS':
            self._run_TOPSIS(criterion_weights, save, file_path)
        elif method.upper() == 'ELECTRE':
            self._run_ELECTRE()
        else:
            raise ValueError('`method` can only be "TOPSIS" OR "ELECTRE", '
                             f'not {method}.')


    def _run_TOPSIS(self, criterion_weights=None, save=False, file_path=''):
        cr_wt = self.criterion_weights if criterion_weights is None else criterion_weights
        ind_type = self.indicator_type
        rev_ind_type = np.ones_like(ind_type) - ind_type
        ind_wt = self.indicator_weights

        tech_scores = self.tech_scores
        tech_scores_a = tech_scores.values # a for array
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
        criteria = ['T', 'RR', 'Env', 'Econ', 'S']
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
        scores, ranks, winners = [], [], []
        columns = self.alt_names
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
            rank = (num_alt+1) - stats.rankdata(score).astype(int)
            winner = columns.loc[np.where(rank==1)]
            scores.append(score)
            ranks.append(rank)
            winners.append(winner.values.item())

        score_df = pd.DataFrame(scores, columns=columns)
        rank_df = pd.DataFrame(ranks, columns=columns)
        winner_df = pd.DataFrame(winners, columns=['Winner'])
        pre_df = pd.DataFrame({'Ratio': cr_wt.Ratio, 'Description': cr_wt.Description},
                              index=score_df.index)

        score_df = pd.concat([pre_df, score_df], axis=1).reset_index(drop=True)
        rank_df = pd.concat([pre_df, rank_df], axis=1).reset_index(drop=True)
        winner_df = pd.concat([pre_df, winner_df], axis=1).reset_index(drop=True)

        self._perform_socres = score_df
        self._ranks = rank_df
        self._winners = winner_df

        if save:
            if not os.path.isdir(results_path):
                os.mkdir(results_path)
            file_path = file_path if file_path else os.path.join(results_path, 'AHP_TOPSIS.xlsx')
            with pd.ExcelWriter(file_path) as writer:
                winner_df.to_excel(writer, sheet_name='Winner')
                score_df.to_excel(writer, sheet_name='Score')
                rank_df.to_excel(writer, sheet_name='Rank')


    def _run_ELECTRE(self, criterion_weights=None):
        '''NOT READY YET.'''
        raise ValueError('Method not ready yet.')


    def __repr__(self):
        alts = ', '.join(self.alt_names)
        return f'<MCDA: {alts}>'


    @time_printer
    def run_MCDA_multi_scores(self, criterion_weights=None, tech_score_dct={}):
        '''
        Run MCDA with multiple sets of technology scores.

        Parameters
        ----------
        criterion_weights : :class:`pandas.DataFrame`
            Weight scenarios for the different criteria,
            will use default scenarios if not provided.
        tech_score_dct : dict
            Dict containing the technology scores for all criteria.

        Returns
        -------
        score_df_dct : dict
            Dict containing the performance scores, keys are the
            normalized global weights.
        rank_df_dct : dict
            Dict containing the rank of performance scores, keys are the
            normalized global weights.
        winner_df : :class:`pandas.DataFrame`
            MCDA winners. Columns are the global weights, rows are indices for
            the different simulations.
        '''
        if criterion_weights is None:
            criterion_weights = self.criterion_weights

        score_df_dct, rank_df_dct, winner_df_dct = {}, {}, {}
        for n, w in criterion_weights.iterrows():
            scores, ranks, winners = [], [], []

            for k, v in tech_score_dct.items():
                self.tech_scores = v
                self.run_MCDA(criterion_weights=w)
                scores.append(self.perform_scores)
                ranks.append(self.ranks)
                winners.append(self.winners.Winner.values.item())

            name = w.Ratio
            score_df_dct[name] = pd.concat(scores).reset_index(drop=True)
            rank_df_dct[name] = pd.concat(ranks).reset_index(drop=True)
            winner_df_dct[name] = winners

        winner_df = pd.DataFrame.from_dict(winner_df_dct)

        return score_df_dct, rank_df_dct, winner_df


    def correlation_test(self, input_x, input_y, kind,
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

    @property
    def winners(self):
        '''[:class:`pandas.DataFrame`] The alternatives that rank first.'''
        if self._winners is None:
            self.run_MCDA()
        return self._winners