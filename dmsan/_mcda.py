#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making for sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>

    Tori Morgan <vlmorgan@illinois.edu>

    Hannah Lohman <hlohman94@gmail.com>

    Stetson Rowles <stetsonsc@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/DMsan/blob/main/LICENSE.txt
for license details.
'''


# %%

import os, numpy as np, pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from qsdsan.utils import time_printer
from . import data_path

__all__ = ('MCDA',)

supported_criteria = ('T', 'RR', 'Env', 'Econ', 'S')
single_cr_df = pd.DataFrame(
    data=np.diag([1]*len(supported_criteria)),
    columns=supported_criteria, dtype='float')


# %%

class MCDA:
    '''
    Class for performing multi-criteria decision analysis.

    Parameters
    ----------
    file_path : str
        Path for the Excel data file containing information related to
        criteria weighing, default path (and file) will be used if not provided.
    alt_names : Iterable(str)
        Names of the alternatives under consideration.
    systems : Iterable(obj)
        Alternative systems to be evaluated in MCDA.
    method : str
        MCDA method, either TOPSIS (technique for order of preference by similarity
        to ideal solution) or ELECTRE (ELimination Et Choice Translating REality).
    indicator_weights : :class:`pandas.DataFrame`
        Calculated weights for indicators in the considered criteria.
    indicator_type : :class:`pandas.DataFrame`
        Columns should be the code of the indicators as used in :class:`DMsan.,
        values should by either "1" (beneficial) or "0" (non-beneficial).
        For beneficial indicators, the higher the indicator score is, the better;
        and vice versa for non-beneficial indicators.
    indicator_scores : :class:`pandas.DataFrame`
        Calculated scores for the alternatives with regard to each indicator.
    criterion_weights : :class:`pandas.DataFrame`
        Weight scenarios for the different criteria,
        will use default scenarios if not provided.

    Examples
    --------
    NOT READY YET.

    '''

    def __init__(self,  file_path='', alt_names=(), systems=(), method='TOPSIS',
                 indicator_type=None, indicator_weights=None, indicator_scores=None,
                 criteria=supported_criteria, criterion_weights=None):
        path = file_path or os.path.join(
            data_path, 'criteria_and_indicators.xlsx')
        file = pd.ExcelFile(path)
        def read_excel(name): return pd.read_excel(
            file, name)  # name is sheet name

        self.alt_names = alt_names
        self.systems = systems
        self.indicator_weights = indicator_weights
        self._default_definitions = defs = read_excel('definitions')
        self.indicator_type = indicator_type if indicator_type is not None \
            else pd.DataFrame({
                defs.variable[i]: defs.category_binary[i] for i in defs.index
            }, index=[0])
        if indicator_weights is not None: self.indicator_weights = indicator_weights
        else: # assume equal indicator weights if not given
            ind_wts = self.indicator_type.copy()
            ind_wts.iloc[0] = 1
            self.indicator_weights = ind_wts
        self._indicator_scores = indicator_scores
        self.criteria = criteria
        self.criterion_weights = criterion_weights if criterion_weights is not None \
            else read_excel('weight_scenarios')
        self.method = method
        self._normalized_indicator_scores = self._criterion_scores = \
            self._performance_scores = self._ranks = self._winners = None

    def update_criterion_weights(self, weights):
        '''
        Format and normalize the given criterion weights for MCDA.
        Note that this is for one set of weights only.

        Parameters
        ----------
        weights : dict(str: float) or Iterable[floats]
            Weights for the different criteria.
            If provided as an Iterable, the order in the `criteria` attr will be assumed.
        '''
        if isinstance(weights, dict):
            weight_df = pd.Series(weights)
        else:
            weight_df = pd.Series(weights, index=self.criteria)
        weight_df /= weight_df.sum()
        weight_df = pd.DataFrame(weight_df).transpose()
        return weight_df

    @staticmethod
    def generate_criterion_weights(wt_scenario_num, criteria=None, seed=3221):
        '''
        Batch-generate criterion weights for uncertainty analysis.

        Parameters
        ----------
        wt_scenario_num : int
            Number of weight scenarios to generate.
        criterion_num : int
            Number of criteria to be considered, default to the number of criteria in MCDA (5).
        seed : int
            Used to create a random number generator for reproducible results.
        '''
        criteria = criteria if criteria is not None else supported_criteria
        criterion_num = len(criteria)
        rng = np.random.default_rng(seed)

        wt_sampler1 = stats.qmc.LatinHypercube(d=1, seed=rng)
        n = int(wt_scenario_num/criterion_num)
        wt1 = wt_sampler1.random(n=n)  # draw from 0 to 1 for one criterion

        wt_sampler4 = stats.qmc.LatinHypercube(d=(criterion_num-1), seed=rng)
        # normalize the rest four based on the first criterion
        wt4 = wt_sampler4.random(n=n)
        tot = wt4.sum(axis=1) / ((np.ones_like(wt1)-wt1).transpose())
        wt4 = wt4.transpose()/np.tile(tot, (wt4.shape[1], 1))

        combined = np.concatenate((wt1.transpose(), wt4)).transpose()

        wts = [combined]
        for num in range(criterion_num-1):
            combined = np.roll(combined, 1, axis=1)
            wts.append(combined)

        weights = np.concatenate(wts).transpose()
        weight_df = pd.DataFrame(weights.transpose(), columns=criteria)

        return weight_df

    @staticmethod
    def plot_criterion_weight_fig(weight_df, path='', **fig_kwargs):
        '''
        Plot all of the criterion weight scenarios.

        Parameters
        ----------
        weight_df : dataframe
            `pandas.DataFrame` with criterion weights.
        path : str
            If provided, the generated figure will be saved to this path.
        fig_kwargs : dict
            Keyword arguments that will be passed to `matplotlib`.
        '''
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(weight_df.transpose(), linewidth=0.5, alpha=0.5, **fig_kwargs)
        ax.set(title='Criterion Weight Scenarios',
               xlim=(0, 4), ylim=(0, 1), ylabel='Criterion Weights',
               xticks=(0, 1, 2, 3, 4),
               xticklabels=weight_df.columns)
        if path: fig.savefig(path)
        return fig

    def run_MCDA(self, method=None, **kwargs):
        '''
        Calculate performance scores using the selected method,
        multiple criterion weights can be considered,
        but only one set of indicator scores are used.

        Parameters
        ----------
        criterion_weights : :class:`pandas.DataFrame`
            Weights for the different criteria, will be defaulted to all of the
            associated criteria in the `criterion_weights` property if left as None.
        method : str
            MCDA method, will use value set in the `method` property if not provided.
        kwargs : dict
            Can assign `criterion_weights`, `indicator_type`, and `indicator_scores`.

        See Also
        --------
        :func:`run_MCDA_multi_scores` for running multiple sets of
        criterion weights and indicator scores.
        '''
        # Update values
        method = method or self.method
        criterion_weights = kwargs.pop('criterion_weights', None)
        if criterion_weights is not None: self.criterion_weights = criterion_weights
        indicator_type = kwargs.pop('indicator_type', None)
        if indicator_type is not None: self.indicator_type = indicator_type
        indicator_scores = kwargs.pop('indicator_scores', None)
        if indicator_scores is not None: self.indicator_scores = indicator_scores

        if method.upper() == 'TOPSIS':
            returned = self._run_TOPSIS(**kwargs)
        elif method.upper() == 'ELECTRE':
            returned = self._run_ELECTRE(**kwargs)
        elif method.upper() == 'AHP':
            returned = self._run_AHP(**kwargs)
        else:
            raise ValueError('`method` can only be "TOPSIS", "ELECTRE", or "AHP", '
                             f'not {method}.')
        return returned

    def _run_TOPSIS(self, **kwargs):
        cr_wt = self.criterion_weights.copy() # make a copy to avoid modifying the original
        # For indicator types, 0 is non-beneficial (want low value) and 1 is beneficial
        ind_type = self.indicator_type.values
        rev_ind_type = np.ones_like(ind_type) - ind_type
        ind_wt = self.indicator_weights
        ind_scores = self.indicator_scores

        ##### Step 1: Normalize indicator scores (vector normalization) #####
        denominator = ((ind_scores**2).sum())**0.5
        norm_ind_scores = self._normalized_indicator_scores = ind_scores/denominator

        ##### Step 2: Normalize criterion weights #####
        N = 1
        if hasattr(cr_wt, 'shape'):
            if len(cr_wt.shape) == 2:  # multiple sets of criterion weights, pd.DataFrame
                N = cr_wt.shape[0]
        if N == 1:  # only 1 set of criterion weights, iterable/pd.Series
            cr_wt = self.update_criterion_weights(cr_wt)
        else:
            cr_wt = cr_wt[[*self.criteria]]
            # normalize the criterion weights
            cr_wt = cr_wt.div(cr_wt.sum(axis=1), 'index')

        ##### Step 3: Calculate normalized indicator scores #####
        # Broadcast the criterion weights to match the number of columns of indicator scores
        criteria = self.criteria
        def get_ind_num(abbr): return len(
            [i for i in ind_wt.columns if i.startswith(abbr)])
        self._ind_num = ind_num = {
            abbr: get_ind_num(abbr) for abbr in criteria}
        cr_wt_arr = np.concatenate(
            [np.tile(cr_wt[i], (ind_num[i], 1)) for i in criteria]).transpose()
        cr_wt = pd.DataFrame(cr_wt_arr, columns=ind_scores.columns)

        # Multiply normalized indicator weights by normalized criterion weights
        norm_wt = cr_wt * ind_wt.values  # multiply by the array for auto-broadcasting

        # Use numpy array to improve speed,
        # a for array
        norm_ind_scores_a = norm_ind_scores.values
        # num of alternatives, num of indicators
        num_alt, num_ind = norm_ind_scores_a.shape
        norm_wt_a = norm_wt.values
        num_cr, num_ind = norm_wt_a.shape  # num of criterion weights, num of indicators

        # Broadcast to the needed shape (add a dimension and repeat num_cr times),
        # b for broadcasted
        norm_ind_scores_b = np.broadcast_to(
            norm_ind_scores_a, (num_cr, num_alt, num_ind))

        # Multiply normalized indicator weights by normalized criterion weights,
        # c for calculated
        norm_ind_scores_c = np.zeros_like(norm_ind_scores_b)
        for i in range(num_cr):
            norm_ind_scores_c[i] = norm_ind_scores_b[i] * norm_wt_a[i]

        ##### Step 4: Calculate the performance scores #####
        # Get ideal best and worst values for each indicator,
        # best would be the max/min for beneficial/non-beneficial indicators
        scores_min = norm_ind_scores_c.min(axis=1)
        scores_max = norm_ind_scores_c.max(axis=1)
        best = (ind_type*scores_max) + (rev_ind_type*scores_min)
        worst = (rev_ind_type*scores_max) + (ind_type*scores_min)

        # Reshape and broadcast,
        # r for reshape
        best_b = np.reshape(best, (num_cr, 1, num_ind))
        best_b = np.tile(best_b, (1, num_alt, 1))
        worst_b = np.reshape(worst, (num_cr, 1, num_ind))
        worst_b = np.tile(worst_b, (1, num_alt, 1))

        # Calculate the Euclidean distance from best and worst,
        # d for distance
        best_d = np.nansum((norm_ind_scores_c-best_b)**2, axis=2) ** 0.5
        worst_d = np.nansum((norm_ind_scores_c-worst_b)**2, axis=2) ** 0.5

        # Calculate performance scores, ranks, and find the winner
        alt_names = self.alt_names
        scores = worst_d / (best_d+worst_d)
        score_df = pd.DataFrame(scores, columns=alt_names)
        rank_df = score_df.rank(axis=1, ascending=False)
        winner_df = rank_df.iloc[:, 0]
        winner_df.name = 'Winner'
        for n, alt in enumerate(alt_names):
            alt_df = rank_df.iloc[:, n]
            winner_df[alt_df == rank_df.min(axis=1)] = alt
        winner_df = pd.DataFrame(winner_df)

        if kwargs.get('update_attr') is not False:
            self._performance_scores = score_df
            self._ranks = rank_df
            self._winners = winner_df
        return score_df, rank_df, winner_df

    def _run_ELECTRE(self, **kwargs):
        '''NOT READY YET.'''
        raise ValueError('Method not ready yet.')

    def _run_AHP(self, **kwargs):
        '''NOT READY YET.'''
        raise ValueError('Method not ready yet.')

    def __repr__(self):
        alts = ', '.join(self.alt_names)
        return f'<MCDA: {alts}>'

    @time_printer
    def run_MCDA_multi_scores(self, criterion_weights=None, method=None,
                              ind_score_dct={}, **kwargs):
        '''
        Run MCDA with multiple sets of criterion weights and indicator scores.

        Parameters
        ----------
        criterion_weights : :class:`pandas.DataFrame`
            Weight scenarios for the different criteria,
            will use default scenarios if not provided.
        method : str
            MCDA method, will use value set in the `method` property if not provided.
        ind_score_dct : dict
            Dict containing the indicator scores for all criteria.

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

        cr_wt = criterion_weights if criterion_weights is not None else self.criterion_weights

        scores, ranks, winners = [], [], []
        for k, v in ind_score_dct.items():
            self.indicator_scores = v
            score_df, rank_df, winner_df = self.run_MCDA(
                criterion_weights=cr_wt, update_attr=kwargs.get('update_attr'))
            scores.append(score_df)
            ranks.append(rank_df)
            winners.append(winner_df)

        weights = cr_wt.values.tolist()
        weights = [':'.join(str(weight).strip('[]').split(', ')) for weight in weights]

        num_ind = len(ind_score_dct)
        score_df_dct, rank_df_dct, winner_df_dct = {}, {}, {}
        num_range = range(num_ind)
        idx = pd.MultiIndex.from_product((num_range, self.alt_names.values))
        for n, name in enumerate(weights):
            score_df = pd.concat([scores[i].iloc[n] for i in num_range])
            score_df.index = idx
            score_df = score_df.unstack()
            rank_df = pd.concat([ranks[i].iloc[n] for i in num_range])
            rank_df.index = idx
            rank_df = rank_df.unstack()
            score_df_dct[name] = score_df
            rank_df_dct[name] = rank_df
            winner_df = pd.concat([winners[i].iloc[n] for i in num_range])
            winner_df.name = name
            winner_df_dct[name] = winner_df

        winner_df = pd.concat(winner_df_dct.values(), axis=1)

        return score_df_dct, rank_df_dct, winner_df

    def calc_criterion_score(self, criterion_weights=None, method=None, ind_score_dct={}):
        '''
        Calculate the score for each criterion by
        setting the criterion weight for that criterion to 1
        while criterion weights for the other criteria to 0.

        Parameters
        ----------
        criterion_weights : :class:`pandas.DataFrame`
            Weights for the different criteria, will be defaulted to all of the
            associated criteria in the `criterion_weights` property if left as None.
        method : str
            MCDA method, will use value set in the `method` property if not provided.
        ind_score_dct : dict
            Dict containing the indicator scores for all criteria,
            if want to calculate the criterion scores for multiple sets of indicator scores.
            If empty, will use the `indicator_scores` attribute to
            get the corresponding criterion scores.
        '''
        if not ind_score_dct:
            score_df, rank_df, winner_df = self.run_MCDA(
                criterion_weights=criterion_weights, method=method,
                file_path=None, update_attr=False)
            return score_df
        else:
            score_df_dct, rank_df_dct, winner_df = self.run_MCDA_multi_scores(
                criterion_weights=criterion_weights, method=method,
                ind_score_dct=ind_score_dct, update_attr=False)
            return score_df_dct

    def correlation_test(self, input_x, input_y, kind,
                         nan_policy='omit', file_path='', **kwargs):
        '''
        Get correlation coefficients between two inputs using `scipy`.

        Parameters
        ----------
        input_x : :class:`pandas.DataFrame`
            The first set of input (typically uncertainty parameters).
        input_y : :class:`pandas.DataFrame`
            The second set of input (typically scores or ranks).
        kind : str
            The type of test to run, can be "Spearman" for Spearman's rho,
            "Pearson" for Pearson's r, "Kendall" for Kendall's tau,
            or "KS" for Kolmogorov–Smirnov's D.

            .. note::
                If running KS test, then input_y should be the ranks (i.e., not scores),
                the x inputs will be divided into two groups - ones that results in
                a given alternative to be ranked first vs. not the first.

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
        :class:`pandas.DataFrame` containing the sensitivity indices and p-values.

        See Also
        --------
        :func:`scipy.stats.spearmanr`

        :func:`scipy.stats.pearsonr`

        :func:`scipy.stats.kendalltau`

        :func:`scipy.stats.kstest`
        '''
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
            if not int(input_y.iloc[0, 0]) == input_y.iloc[0, 0]:
                raise ValueError(
                    'For KS test, `input_y` should be the ranks, not scores.')

            func = stats.kstest
            stats_name = 'D'

            alternative = kwargs.get('alternative') or 'two_sided'
            mode = kwargs.get('mode') or 'auto'
        else:
            raise ValueError('kind can only be "Spearman", "Pearson", '
                             f'or "Kendall", not "{kind}".')

        X = np.array(input_x)
        X = X.reshape((X.shape[0], 1)) if len(X.shape) == 1 else X
        XT = X.T
        cols_x = input_x.columns if hasattr(input_x, 'columns') else range(X.shape[1])
        Y = np.array(input_y)
        Y = Y.reshape((Y.shape[0], 1)) if len(Y.shape) == 1 else Y
        YT = Y.T
        cols_y = input_y.columns if hasattr(input_y, 'columns') else range(Y.shape[1])

        # Results, the shape is
        # the num of parameters, 2 (indices & p)*num of y inputs
        if name != 'ks':
            results = np.concatenate([
                np.array([func(x, y, **kwargs) for x in XT])
                for y in YT
                ])
        else:
            num_param = X.shape[1]
            results = []
            for y in YT:
                i_win, i_lose = X[y==1], X[y!=1]
                if len(i_win) == 0 or len(i_lose) == 0:
                    result = np.full((num_param, 2), None)
                else:
                    i_winT = i_win.T
                    i_loseT = i_lose.T
                    result = np.array([
                        func(i_winT[n], i_loseT[n], alternative=alternative, mode=mode, **kwargs)
                        for n in range(i_winT.shape[0])
                        ])
                results.append(result)
            results = np.concatenate(results, axis=1)

        # Parameter names
        columns0 = pd.MultiIndex.from_arrays([('',), ('Parameter',)], names=('Y', 'Stats'))
        df0 = pd.DataFrame(data=cols_x, columns=columns0)
        # Result header
        columns1 = pd.MultiIndex.from_product((cols_y, (stats_name, 'p-value')))
        df1 = pd.DataFrame(data=results, columns=columns1)
        df = pd.concat((df0, df1), axis=1)

        if file_path: df.to_csv(file_path)

        return df

    @property
    def alt_names(self):
        '''[:class:`pandas.Series`] Names of the alternative systems under consideration.'''
        return self._alt_names

    @alt_names.setter
    def alt_names(self, i):
        self._alt_names = pd.Series(i)

    @property
    def criteria(self):
        '''[tuple(str)] All criteria considered in MCDA.'''
        return self._criteria
    @criteria.setter
    def criteria(self, i):
        self._criteria = i

    @property
    def systems(self):
        '''[list] Alternative systems evaluated by MCDA.'''
        return self._systems
    @systems.setter
    def systems(self, i):
        self._systems = list(i)

    @property
    def indicator_scores(self):
        '''[:class:`pandas.DataFrame`] Raw indicator scores (i.e., no need to be normalized).'''
        return self._indicator_scores

    @indicator_scores.setter
    def indicator_scores(self, i):
        self._indicator_scores = i
        self.run_MCDA()

    @property
    def normalized_indicator_scores(self):
        '''[:class:`pandas.DataFrame`] Indicator scores normalized based on their scales.'''
        if self._normalized_indicator_scores is None:
            try: self.run_MCDA()
            except: pass
        return self._normalized_indicator_scores

    @property
    def criterion_scores(self):
        '''[dict or :class:`pandas.DataFrame`] Criterion scores.'''
        if self._criterion_scores is None:
            self._criterion_scores = \
                self.calc_criterion_score(criterion_weights=single_cr_df)
        return self._criterion_scores

    @property
    def performance_scores(self):
        '''[:class:`pandas.DataFrame`] Calculated performance scores.'''
        if self._performance_scores is None:
            try:
                self.run_MCDA()
            except:
                pass
        return self._performance_scores

    @property
    def ranks(self):
        '''[:class:`pandas.DataFrame`] Calculated ranks.'''
        if self._ranks is None:
            try:
                self.run_MCDA()
            except:
                pass
        return self._ranks

    @property
    def winners(self):
        '''[:class:`pandas.DataFrame`] The alternatives that rank first.'''
        if self._winners is None:
            try:
                self.run_MCDA()
            except:
                pass
        return self._winners