#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors:
    Tori Morgan <vlmorgan@illinois.edu>,
    Hannah Lohman <hlohman94@gmail.com>,
    Yalin Li <mailto.yalin.li@gmail.com>,
    Stetson Rowles <stetsonsc@gmail.com>

This model is developed to assist sanitation system research, development, and
deployment.
"""


# %%

import numpy as np
import pandas as pd
from . import Location

__all__ = ('AHP',)


class AHP:
    '''
    Determine the local weights of indicators in technical, resource recovery,
    economic, environmental, and social criteria using the
    analytic hierarchy process (AHP).

    Parameters
    ----------
    file_path : str
        Path for the Excel data file containing contextual parameters for
        different countries, default path (and file) will be used if not provided.
    location_name : str
        Name of the location by country.
    init_weights : dict(str, float)
        Dictionary of the initial indicator weights, the keys should be the indicator name
        and the values should be the indicator weight.
        Default indicator weights (for the specific country) will be used
        if not given in the dict.
    num_alt : int
        Number of alternatives to be evaluated.
    na_default : float
        Default values when an indicator is empty (i.e., N/A).
    random_index : dict
        Random indices corresponding to the number of the sub-criteria,
        default ones will be used if not given.

    Examples
    --------
    NOT READY YET.
    '''


    def __init__(self, file_path='', location_name='Uganda', init_weights={},
                 num_alt=3, na_default=0.00001, random_index={}):
        # Convert location to match the database
        self.set_location(location_name=location_name, file_path=file_path)
        self.num_alt = int(num_alt)
        self.na_default = na_default
        if not random_index:
            self.random_index = {
                3: 0.58,
                4: 0.90,
                5: 1.12,
                6: 1.24,
                7: 1.32,
                8: 1.41,
                9: 1.45,
                10: 1.49,
                11: 1.51,
                12: 1.54
                }
        if random_index:
            self.random_index.update(random_index)
        # Set initial weights for different criteria
        self._init_weights = self._get_default_init_weights()
        self.init_weights = init_weights
        self.get_indicator_weights()

    def update_init_weights(self, init_weights={}):
        init_weights = init_weights or {}
        self._init_weights.update(init_weights)
        weights = self._init_weights
        init_weights_df = pd.DataFrame(list(weights.values())).transpose()
        get_ind_range = lambda abbr: range(len([i for i in weights.keys() if i.startswith(abbr)]))
        init_weights_df.columns = [
            *[f'T{i+1}' for i in get_ind_range('T')],
            *[f'RR{i+1}' for i in get_ind_range('RR')],
            *[f'Env{i+1}' for i in get_ind_range('Env')],
            *[f'Econ{i+1}' for i in get_ind_range('Econ')],
            *[f'S{i+1}' for i in get_ind_range('S')],
            ]
        self._init_weights_df = init_weights_df


    def _get_val(self, df, col='Value'):
        '''Util function for retrieving data.'''
        return df.loc[self.location.location_name, col]

    def _get_default_init_weights(self):
        ''' Set default initial indicator weights.'''
        ##### Technical indicators #####
        get_val = self._get_val
        location = self.location
        weights = {}

        # Sub-criteria: Resilience
        # Local Weight Indicator T1: Extent of training
        # related to how much training is available to train users and personnel
        weights['T1'] = 100 - (get_val(location.training)/7*100)

        # Local Weight Indicator T2: Population with access to improved sanitation
        # related to how available improved sanitation is in the region in case a system fails
        weights['T2'] = 100 - get_val(location.sanitation_availability, 'Value - Improved Sanitation')

        # Sub-criteria: Feasibility
        # Local Weight Indicator T3: Accessibility to technology
        # related to how easily the region can access technology
        weights['T3'] = 100-(get_val(location.tech_absorption)/7*100)

        # Local Weight Indicator T4: Transportation infrastructure
        # related to the quality of transportation infrastructure for transport of waste
        weights['T4'] = 100-(get_val(location.road_quality)/7*100)

        # Local Weight Indicator T5: Construction skills available
        # related to the construction expertise available
        weights['T5'] = 100 - (get_val(location.construction)/40.5*100)

        # Local Weight Indicator T6: O&M expertise available
        # related to the O&M expertise available
        weights['T6'] = 100-(get_val(location.OM_expertise)/7*100)

        # Local Weight Indicator T7: Population growth trajectory
        # related to the population flexibility
        weights['T7'] = get_val(location.pop_growth)/4.47*100

        # Local Weight Indicator T8:
        # related to the grid-electricity flexibility
        weights['T8'] = get_val(location.electricity_blackouts)/75.2*100

        # Local Weight Indicator T9:
        # related to the drought flexibility
        weights['T9'] = get_val(location.water_stress)/4.82*100

        ##### Resource recovery #####
        # Local Weight Indicator RR1:
        # related to the water stress (Water Recovery)
        weights['RR1'] = weights['T9']

        # Local Weight Indicator RR2:
        # related to nitrogen (N) fertilizer fulfillment (Nutrient Recovery)
        weights['RR2'] = (1-(get_val(location.n_fertilizer_fulfillment)/100))*100

        # Local Weight Indicator RR3:
        # related to phosphorus (P) fertilizer fulfillment (Nutrient Recovery)
        weights['RR3'] = (1-(get_val(location.p_fertilizer_fulfillment)/100))*100

        # Local Weight Indicator RR4:
        # related to potassium (K) fertilizer fulfillment (Nutrient Recovery)
        weights['RR4'] = (1-(get_val(location.k_fertilizer_fulfillment)/100))*100

        # Local Weight Indicator RR5:
        # related to renewable energy consumption (Energy Recovery)
        weights['RR5'] = get_val(location.renewable_energy)

        # Local Weight Indicator RR6:
        # related to infrastructure quality (Supply Chain Infrastructure)
        weights['RR6'] = (1-(get_val(location.infrastructure)/7))*100

        ##### Environmental #####
        # Env1: ecosystem quality (LCA)
        # Env2: human health (LCA)
        # Env3: resource depletion (LCA)
        val = 1/self.num_alt
        for i in range(3): weights[f'Env{i+1}'] = val

        ##### Economic #####
        weights['Econ1'] = 1 # only one for the net cost

        ##### Social #####
        X = self.na_default

        # Sub-criteria: Job Creation
        # Local Weight Indicator S1: Unemployment
        # related to the unemployment rate
        weights['S1'] = get_val(location.unemployment_rate)/28.74*100

        # Local Weight Indicator S2: High paying jobs
        # related to the need for higher paying jobs
        weights['S2'] = get_val(location.high_pay_jobs)/94.3*100

        # Sub-criteria: End-user acceptability, S3-S7
        # !!! Input community preference
        # Local Weight Indicator S3: Disposal convenience preference for user
        # related to the preference for disposal requirements on the user end
        # if management is responsible for disposal or if the community did not describe
        # this indicator as important, then insert X
        # 0 being low preference to frequency of disposal to 100 being high preference for frequency of disposal
        # ## Specific to Bwaise example: community did not mention disposal as affecting their acceptability ##
        weights['S3'] = X

        # !!! Input community preference
        # Local Weight Indicator S4: Cleaning preference
        # related to the preference for cleaning requirements
        # 0 being low preference to frequency of cleaning to 100 being high preference for frequency of cleaning
        weights['S4'] = 44

        # !!! Input community preference
        # Local Weight Indicator S5: Privacy preference
        # related to the preference for privacy (# of households sharing a system)
        # 0 being low preference for privacy to 100 being high preference for privacy
        weights['S5'] = 47

        # !!! Input community preference
        # Local Weight Indicator S6: Odor preference
        # related to the preference of odor with
        # 0 being low preference for odor to 100 being high preference for odor
        weights['S6'] = 22

        # !!! Input community preference
        # Local Weight Indicator S7: Security preference
        # related to the preference of security with
        # 0 being low preference for security to 100 being high preference for security
        # ## Specific to Bwaise example: community did not mention disposal as affecting their acceptability ##
        weights['S7'] = 24

        # Sub-criteria: Management Acceptability, S8 & S9
        # !!! Input management (i.e., landlord) preference
        # Local Weight Indicator S8: Disposal convenience preference
        # related to the preference for disposal requirements
        # 0 being low importance to frequency of disposal to 100 being high importance for frequency of disposal
        # ## Specific to Bwaise example: the sanitation system is controlled by the end-user, not the landlord ##
        weights['S8'] = X

        # ## Input management preference ##
        # Local Weight Indicator S9: Cleaning preference
        # related to the preference for cleaning requirements
        # 0 being low importance to frequency of cleaning to 100 being high importance for frequency of cleaning
        weights['S9'] = X

        return weights


    def __repr__(self):
        return f'<AHP: {self.location.location_name}>'


    def get_indicator_weights(self, init_weights=None, return_results=False):
        '''Analytic hierarchy process (AHP) to determine indicators weights.'''
        RI = self.random_index
        norm_weights = self.norm_weights = {} # sub-criteria weights
        CRs = self.CRs = {} # consistency ratio
        if not init_weights:
            init_weights_df = self.init_weights_df
        if isinstance(init_weights, dict) or init_weights is None:
            self.update_init_weights(init_weights)
            init_weights_df = self.init_weights_df
        else: # assume to be a dataframe
            init_weights_df = init_weights

        for indicator, weights in init_weights_df.items():
            num = len(weights)
            index = [f'{indicator}{i+1}' for i in range(num)]

            if num < 3: # skip ones that does not have random index (RI)
                norm_weights[indicator] = pd.DataFrame([1], index=index).transpose()
                continue

            # Step 1: Assign criteria weights in array
            # "A" stands for array
            A1 = [[i]*num for i in weights] # e.g., [[T1, T1], [T2, T2]]
            A2 = [weights] * num # e.g., [[T1, T2], [T1, T2]]
            A = np.array(A1) / np.array(A2) # e.g., [[T1/T1, T1/T2], [T2/T1, T2/T2]]

            # Step 2: Sum the columns
            A_col_sum = A.sum(0)

            # Step 3: Normalize the array
            A_norm = A / A_col_sum

            # Step 4: Calculate criteria weights by finding the row averages
            A_norm_sum = A_norm.sum(1)
            A_norm_avg = A_norm_sum / num
            norm_weights[indicator] = \
                pd.DataFrame(A_norm_avg, index=index).transpose()

            # Step 5 Find the Consistency ratio
            # Step 5a: Calculate the weighted array by multiplying the array by the criteria weight
            # Step 5b: Sum the rows of the weighted matrix to find the weighted sum value
            A_weighted = np.matmul(A.T, A_norm_sum.T)
            # Step 5c: divide the weighted sum value by the criteria weights
            ratio = A_weighted / A_norm_sum
            # Step 5d: Find the consistency index (CI) by calculating (delta_max - n)/(n-1)
            delta_max = ratio.sum() / num
            CI = (delta_max - num) / (num - 1)
            # Step 5e: Find the consistency ratio (CR) by dividing CI by RI,
            # if CR < 0.1 then our matrix is consistent
            CRs[indicator] = CI / RI[num]

        self._norm_weights_df = pd.concat([i for i in norm_weights.values()], axis=1)

        if return_results:
            return self.norm_weights_df

    def set_location(self, location_name, file_path=''):
        self._location = Location(file_path=file_path, location_name=location_name)

    @property
    def location(self):
        '''[:class:`~.Location`] Selected location of interest.'''
        return self._location

    @property
    def init_weights(self):
        '''[dict] Initial indicator weights.'''
        return self._init_weights
    @init_weights.setter
    def init_weights(self, i):
        if isinstance(i, dict): self.update_init_weights(i)
        else: raise ValueError('`init_weights` should be a dict, '
                               f'not {type(i).__name__}.')

    @property
    def init_weights_df(self):
        '''[:class:`pandas.DataFrame`] Initial indicator weights.'''
        return self._init_weights_df

    @property
    def norm_weights_df(self):
        '''[:class:`pandas.DataFrame`] Normalized indicator weights.'''
        return self._norm_weights_df