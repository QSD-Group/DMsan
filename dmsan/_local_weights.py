#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 11:40:50 2021

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

import numpy as np
import pandas as pd
import os
from scipy.stats import rankdata
from . import Location, data_path

# Data files associated with the model
result_path = os.path.join(os.path.dirname(__file__), 'results')
data_path_tech_scores = os.path.join(data_path, 'technology_scores.xlsx')
data_path_weight_scenarios = os.path.join(data_path, 'criteria_weight_scenarios.xlsx')



# %%

class LocalWeights:
    '''
    Contains the information for local weights of technical,
    resource recovery, economic, environmental, and social criteria.

    Parameters
    ----------
    location_name : str
        Name of the location by country.
    num_alt : int
        Default values when an indicator is empty (i.e., N/A).
    num_alt : int
        Number of alternatives to be evaluated.
    RI : dict
        Random indices corresponding to the number of the sub-criteria,
        default ones will be used if not given.

    Examples
    --------
    NOT READY YET.

    '''
    def __init__(self, location_name='Uganda', num_alt=3, na_default=0.00001,
                 random_index={}):
        # Convert location to match the database
        self.location = Location(location_name)
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

        # Initiate the dict
        self.init_weights = {}

        # Set initial weights for different criteria
        self._set_T_weights()
        self._set_RR_weights()
        self._set_Econ_weights()
        self._set_Env_weights()
        self._set_S_weights()

    def _get_val(self, df, col='Value'):
        '''Util function for retrieving data.'''
        return df.loc[self.location.location_name, col]

    def _set_init_T_weights(self):
        '''Set initial weights for technical criteria.'''
        get_val = self._get_val
        location = self.location
        self.init_weights['technical'] = weights = []

        # Sub-criteria: Resilience
        # Local Weight Indicator T1: Extent of training
        # related to how much training is available to train users and personnel
        weights.append(100 - (get_val(location.training)/7*100))

        # Local Weight Indicator T2: Population with access to imporved sanitation
        # related to how available improved sanitation is in the region in case a system fails
        weights.append(100 - get_val(location.sanitation_availability, 'Value - Improved Sanitation'))

        # Sub-criteria: Feasibility
        # Local Weight Indicator T3: Accessibility to technology
        # related to how easily the region can access technology
        weights.append(100-(get_val(location.tech_absorption)/7*100))

        # Local Weight Indicator T4: Transportation infrastructure
        # related to the quality of transportation infrastructure for transport of waste
        weights.append(100-(get_val(location.road_quality)/7*100))

        # Local Weight Indicator T5: Construction skills available
        # related to the construction expertise available
        weights.append(100 - (get_val(location.construction)/40.5*100))

        # Local Weight Indicator T6: O&M expertise available
        # related to the O&M expertise available
        weights.append(100-(get_val(location.OM_expertise)/7*100))

        # Local Weight Indicator T7: Population growth trajectory
        # related to the population flexibility
        weights.append(get_val(location.pop_growth)/4.5*100)

        # Local Weight Indicator T8:
        # related to the grid-electricity flexibility
        weights.append(100-(get_val(location.electricity_blackouts)/72.5*100))

        # Local Weight Indicator T9:
        # related to the drought flexibility
        weights.append(100-(get_val(location.water_stress)/4.82*100))


    def _set_init_RR_weights(self):
        '''Set initial weights for resource recovery criteria.'''
        get_val = self._get_val
        location = self.location
        self.init_weights['resource_recovery'] = weights = []

        # Local Weight Indicator RR1:
        # related to the water stress (Water Recovery)
        weights.append(self.init_weights['technical'][-1])

        # Local Weight Indicator RR2:
        # related to nitrogen (N) fertilizer fulfillment (Nutrient Recovery)
        weights.append(1 - (get_val(location.n_fertilizer_fulfillment)/100)) * 100

        # Local Weight Indicator RR3:
        # related to phosphorus (P) fertilizer fulfillment (Nutrient Recovery)
        weights.append(1 - (get_val(location.p_fertilizer_fulfillment)/100)) * 100

        # Local Weight Indicator RR4:
        # related to potassium (K) fertilizer fulfillment (Nutrient Recovery)
        weights.append(1 - (get_val(location.k_fertilizer_fulfillment)/100)) * 100

        # Local Weight Indicator RR5:
        # related to renewable energy consumption (Energy Recovery)
        weights.append(1 - (get_val(location.renewable_energy)/100)) * 100

        # Local Weight Indicator RR6:
        # related to infrastructure quality (Supply Chain Infrastructure)
        weights.append(1 - (get_val(location.infrastructure)/7)) * 100

    def _set_init_Econ_weights(self):
        '''Set initial weights for economic criteria.'''
        self.init_weights['economic'] = [1] # only one for the net cost

    def _set_init_Env_weights(self):
        '''Set initial weights for economic criteria.'''
        # Local Weight Indicator
        # Env1: ecosystem quality (LCA)
        # Env2: human health (LCA)
        # Env3: resource depletion (LCA)
        self.init_weights['environmental'] = [1/self.num_alt] * self.num_alt


    def _set_init_S_weights(self):
        '''Set initial weights for social criteria.'''
        get_val = self._get_val
        location = self.location
        X = self.na_default
        self.init_weights['social'] = weights = []

        # Sub-criteria: Job Creation
        # Local Weight Indicator S1: Unemployment
        # related to the unemployment rate
        weights.append(get_val(location.unemployment_rate)/28.74*100)

        # Local Weight Indicator S2: High paying jobs
        # related to the need for higher paying jobs
        weights.append(get_val(location.high_pay_jobs)/94.3*100)

        # Sub-criteria: End-user acceptability, S3-S7
        # !!! Input community preference
        # Local Weight Indicator S3: Disposal convenience preference for user
        # related to the preference for disposal requirements on the user end
        # if management is responsible for disposal or if the community did not describe
        # this indicator as important, then insert X
        # 0 being low preference to frequency of disposal to 100 being high preference for frequency of disposal
        # ## Specific to Bwaise example: community did not mention disposal as affecting their acceptability ##
        weights.append(X)

        # !!! Input community preference
        # Local Weight Indicator S4: Cleaning preference
        # related to the preference for cleaning requirements
        # 0 being low preference to frequency of cleaning to 100 being high preference for frequency of cleaning
        weights.append(44)

        # !!! Input community preference
        # Local Weight Indicator S5: Privacy preference
        # related to the preference for privacy (# of households sharing a system)
        # 0 being low preference for privacy to 100 being high preference for privacy
        weights.append(47)

        # !!! Input community preference
        # Local Weight Indicator S6: Odor preference
        # related to the preference of odor with
        # 0 being low preference for odor to 100 being high preference for odor
        weights.append(22)

        # !!! Input community preference
        # Local Weight Indicator S7: Security preference
        # related to the preference of security with
        # 0 being low preference for secutiy to 100 being high preference for odor
        # ## Specific to Bwaise example: community did not mention disposal as affecting their acceptability ##
        weights.append(X)

        # Sub-criteria: Management Acceptability, S8 & S9
        # !!! Input management (i.e., landlord) preference
        # Local Weight Indicator S8: Disposal convenience preference
        # related to the preference for disposal requirements
        # 0 being low importance to frequency of disposal to 100 being high importance for frequency of disposal
        # ## Specific to Bwaise example: the sanitation system is controlled by the end-usr, not landlord ##
        weights.append(X)

        # ## Input management preference ##
        # Local Weight Indicator S9: Cleaning preference
        # related to the preference for cleaning requirements
        # 0 being low importance to frequency of cleaning to 100 being high importance for frequency of cleaning
        weights.append(X)


    def get_AHP_weights(self):
        '''Analytic hierarchy process (AHP) to determine sub-criteria weights.'''
        RI = self.random_index

        for weights in self.init_weights.values():
            if len(weights) < 3:
                continue

            # Step 1: Assign criteria weights in array
            num = len(weights)
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
            CR = CI / RI[num]

    #!!! PAUSED, AHP finished, now need to add TOPSIS and ELECTRE