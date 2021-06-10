# -*- coding: utf-8 -*-
"""
Modified on Tue May 4

@author:
    torimorgan <vlmorgan@illinois.edu>,
    Hannah Lohman <hlohman94@gmail.com>,
    stetsonrowles,
    Yalin Li <zoe.yalin.li@gmail.com>

test change
"""

                            ## MCDA Model ##
# from __future__ import division #floating point division in Py2x
import numpy as np
# import math
import pandas as pd
import country_converter as coco
import os
from .env import get_LCA_baseline
data_path = os.path.abspath(os.path.dirname('location.xlsx'))

# class MCDA:

# Step 1: Identify Location

# ## Input location by country in the brackets ##
location = ['Uganda']

# converting location to match the database
location = coco.convert(names=location, to='name_short')

# manual inputs defined here

X = 0.00001

# Criteria: Social
# Sub-criteria: End-user acceptability
# ## Input community preference ##
# Local Weight Indicator S3: Disposal convenience preference for user
# relates to the preference for disposal requirements on the user end
# if management is responsible for disposal, then insert X
# 0 being low preference to frequency of disposal to 100 being high preference for frequency of disposal
s3 = X

# ## Input community preference ##
# Local Weight Indicator S4: Cleaning preference
# relates to the preference for cleaning requirements
# 0 being low preference to frequency of cleaning to 100 being high preference for frequency of cleaning
s4 = 44

# ## Input community preference ##
# Local Weight Indicator S5: Privacy preference
# relates to the preference for privacy (# of households sharing a system)
# 0 being low preference for privacy to 100 being high preference for privacy
s5 = 47

# ## Input community preference ##
# Local Weight Indicator S6: Odor preference
# relates to the preference of odor with
# 0 being low preference for odor to 100 being high preference for odor
s6 = 22

# ## Input community preference ##
# Local Weight Indicator S7: Noise preference
# relates to the preference of noise with
# 0 being low preference for odor to 100 being high preference for noise
s7 = X

# ## Input community preference ##
# Local Weight Indicator S8: PPE preference/familiarity
# relates to the preference of PPE with
# 0 being low importance for PPE to 100 being high importance for PPE
s8 = X

# ## Input community preference ##
# Local Weight Indicator S9: Security preference
# relates to the preference of security with
# 0 being low preference for secutiy to 100 being high preference for odor
s9 = X

# Sub-criteria: Management Acceptability
# ## Input management (i.e., landlord) preference ##
# Local Weight Indicator S10: Disposal convenience preference
# relates to the preference for disposal requirements
# 0 being low importance to frequency of disposal to 100 being high importance for frequency of disposal
s10 = X

# ## Input management preference ##
# Local Weight Indicator S11: Cleaning preference
# relates to the preference for cleaning requirements
# 0 being low importance to frequency of cleaning to 100 being high importance for frequency of cleaning
s11 = X

# ## Input management preference ##
# Local Weight Indicator S12: PPE preference/familiarity
# relates to the preference of PPE with
# 0 being low importance for PPE to 100 being high importance for PPE
s12 = X


                            # Local Weights #
# Criteria: Technical
# Sub-criteria: Resilience
# Local Weight Indicator T1: Extent of training
# relates to how much training is available to train users and personnel
training = pd.read_excel(data_path+'/location.xlsx', sheet_name='ExtentStaffTraining', index_col='Country')
t1 = (training.loc[location, 'Value'])
T1 = (100 - (t1/7*100))

# Local Weight Indicator T2: Population with access to imporved sanitation
# relates to how available improved sanitation is in the region in case a system fails
sanitation_availability = pd.read_excel(data_path+'/location.xlsx', sheet_name='Sanitation', index_col='Country')
t2 = (sanitation_availability.loc[location, 'Value - Improved Sanitation'])
T2 = (100 - t2)

# Sub-criteria: Feasibility
# Local Weight Indicator T3: Accessibility to technology
# relates to how easily the region can access technology
tech_absorption = pd.read_excel(data_path+'/location.xlsx', sheet_name='TechAbsorption', index_col='Country')
t3 = (tech_absorption.loc[location, 'Value'])
T3 = (100-(t3/7*100))

# Local Weight Indicator T4: Transportation infrastructure
# relates to the quality of transportation infrastructure for transport of waste
road_quality = pd.read_excel(data_path+'/location.xlsx', sheet_name='RoadQuality', index_col='Country')
t4 = (road_quality.loc[location, 'Value'])
T4 = (100-(t4/7*100))

# Local Weight Indicator T5: Construction skills available
# relates to the construction expertise available
construction = pd.read_excel(data_path+'/location.xlsx', sheet_name='Construction', index_col='Country')
t5 = (construction.loc[location, 'Value'])
T5 = (100 - (t5/40.5*100))

# Local Weight Indicator T6: O&M expertise available
# relates to the O&M expertise available
OM_expertise = pd.read_excel(data_path+'/location.xlsx', sheet_name='AvailableScientistsEngineers', index_col='Country')
t6 = (OM_expertise.loc[location, 'Value'])
T6 = (100-(t6/7*100))

# Local Weight Indicator T7: Population growth trajectory
# relates to the population flexibility
pop_growth = pd.read_excel(data_path+'/location.xlsx', sheet_name='PopGrowth', index_col='Country')
t7 = (pop_growth.loc[location, 'Value'])
T7 = (t7/4.5*100)

# Local Weight Indicator T8:
# relates to the climate risk
climate_risk = pd.read_excel(data_path+'/location.xlsx', sheet_name='ClimateRiskIndex', index_col='Country')
t8 = (climate_risk.loc[location, 'Value'])
T8 = (100-(t8/118*100))

# Local Weight Indicator T9:
# relates to the temperature anomalies
temperature_anomalies = pd.read_excel(data_path+'/location.xlsx', sheet_name='TemperatureAnomalies',
                                      index_col='Country')
t9 = (temperature_anomalies.loc[location, 'Value'])
T9 = (100-(t9/3.6*100))

# Local Weight Indicator T10:
# relates to the water stress
water_stress = pd.read_excel(data_path+'/location.xlsx', sheet_name='WaterStress', index_col='Country')
t10 = (water_stress.loc[location, 'Value'])
T10 = (100-(t10/4.82*100))


#Criteria: Environmental
#!!! Yalin, can you add the exposan here and have it in a format similar to the other criteria
# I assume LCA1-3 are for system A-C? I only include the baseline values below,
# since I think we want to use harmonized assumptions for uncertainty analysis
lca_baseline_dct = get_LCA_baseline()
# Only use the hierarchist perspective


lca_baseline_dct = get_LCA_baseline()
# Only use the hierarchist perspective
hierarchist_dct = {'sysA': {}, 'sysB': {}, 'sysC': {}}
for sys_ID, results in lca_baseline_dct.items():
    for ind, value in results.items():
        if ind.startswith('H_'): # change this to 'I_' for individualist or 'E_' for egalitarian
            hierarchist_dct[sys_ID][ind] = value

# Because of the bug in this script it won't run, but here's what you should get,
# note that since each time you need to run the system to retrieve those values,
# it takes some time to get the results
# {'sysA': {'H_EcosystemQuality_Total': -371.04045858430715,
#   'H_HumanHealth_Total': 0.20460887746066936,
#   'H_Resources_Total': 0.2457366918321932},
#  'sysB': {'H_EcosystemQuality_Total': -3871.307383573435,
#   'H_HumanHealth_Total': -0.17709372298736473,
#   'H_Resources_Total': -0.2942841657695248},
#  'sysC': {'H_EcosystemQuality_Total': -1114.683613325984,
#   'H_HumanHealth_Total': 0.41160079447096,
#   'H_Resources_Total': 1.1398505496550142}}

# Sub-criteria: Resource Recovery Potential

# Local Weight Indicator Env1:
# relates to the water stress (Water Recovery)

Env1 = T10

# Local Weight Indicator Env2:
# relates to nitrogen (N) fertilizer fulfillment (Nutrient Recovery)

n_fertilizer_fulfillment = pd.read_excel(data_path+'/location.xlsx', sheet_name='NFertilizerFulfillment',
                                         index_col='Country')
env2 = (n_fertilizer_fulfillment.loc[location, 'Value'])
Env2 = (1 - (env2/100)) * 100

# Local Weight Indicator Env3:
# relates to phosphorus (P) fertilizer fulfillment (Nutrient Recovery)

p_fertilizer_fulfillment = pd.read_excel(data_path+'/location.xlsx', sheet_name='PFertilizerFulfillment',
                                         index_col='Country')
env3 = (p_fertilizer_fulfillment.loc[location, 'Value'])
Env3 = (1 - (env3/100)) * 100

# Local Weight Indicator Env4:
# relates to potassium (K) fertilizer fulfillment (Nutrient Recovery)

k_fertilizer_fulfillment = pd.read_excel(data_path+'/location.xlsx', sheet_name='KFertilizerFulfillment',
                                         index_col='Country')
env4 = (k_fertilizer_fulfillment.loc[location, 'Value'])
Env4 = (1 - (env4/100)) * 100

# Local Weight Indicator Env5:
# relates to renewable energy consumption (Energy Recovery)

renewable_energy = pd.read_excel(data_path+'/location.xlsx', sheet_name='RenewableEnergyConsumption',
                                 index_col='Country')
env5 = (renewable_energy.loc[location, 'Value'])
Env5 = (1 - (env5/100)) * 100

# Local Weight Indicator Env6:
# relates to infrastructure quality (Supply Chain Infrastructure)

infrastructure = pd.read_excel(data_path+'/location.xlsx', sheet_name='InfrastructureQuality',
                                         index_col='Country')
env6 = (infrastructure.loc[location, 'Value'])
Env6 = (1 - (env6/7)) * 100

# Sub-criteria: LCA
# !!!!! #Yalin - see the separate env.py

# Criteria: Social
# Sub-criteria: Job Creation
# Local Weight Indicator S1: Unemployment
# relates to the unemployment rate
unemployment_rate = pd.read_excel(data_path+'/location.xlsx', sheet_name='UnemploymentTotal', index_col='Country')
s1 = (unemployment_rate.loc[location, 'Value'])
S1 = (s1/28.74*100)

# Local Weight Indicator S2: High paying jobs
# relates to the need for higher paying jobs
high_pay_jobs = pd.read_excel(data_path+'/location.xlsx', sheet_name='HighPayJobRate', index_col='Country')
s2 = (high_pay_jobs.loc[location, 'Value'])
S2 = (s2/94.3*100)

# Sub-criteria
# Local Weight Indicator S3: Disposal convenience preference
# relates to the preference for disposal requirements
# 0 being low importance to frequency of disposal to 100 being high importance for frequency of disposal
S3 = s3

# ## Input community preference ##
# Local Weight Indicator S4: Cleaning preference
# relates to the preference for cleaning requirements
# 0 being low importance to frequency of cleaning to 100 being high importance for frequency of cleaning
S4 = s4

# ## Input community preference ##
# Local Weight Indicator S5: Privacy preference
# relates to the preference for privacy (# of households sharing a system)
# 0 being low importance for privacy to 100 being high importance for privacy
S5 = s5

# ## Input community preference ##
# Local Weight Indicator S6: Odor preference
# relates to the preference of odor with
# 0 being low importance for odor to 100 being high importance for odor
S6 = s6

# ## Input community preference ##
# Local Weight Indicator S7: Noise preference
# relates to the preference of noise with
# 0 being low importance for odor to 100 being high importance for noise
S7 = s7

# ## Input community preference ##
# Local Weight Indicator S8: PPE preference/familiarity
# relates to the preference of PPE with
# 0 being low importance for PPE to 100 being high importance for PPE
S8 = s8

# ## Input community preference ##
# Local Weight Indicator S9: Security preference
# relates to the preference of security with
# 0 being low importance for security to 100 being high importance for odor
S9 = s9

# Sub-criteria: Management Acceptability
# ## Input management (i.e., landlord) preference ##
# Local Weight Indicator S10: Disposal convenience preference
# relates to the preference for disposal requirements
# 0 being low importance to frequency of disposal to 100 being high importance for frequency of disposal
S10 = s10

# ## Input management preference ##
# Local Weight Indicator S11: Cleaning preference
# relates to the preference for cleaning requirements
# 0 being low importance to frequency of cleaning to 100 being high importance for frequency of cleaning
S11 = s11

# ## Input management preference ##
# Local Weight Indicator S12: PPE preference/familiarity
# relates to the preference of PPE with
# 0 being low importance for PPE to 100 being high importance for PPE
S12 = s12

# %%
                    # Tech/System Performance Scores#
# class AHP(MCDA):
#     def main():

# Step 1: Assign criteria weights in matrix

T_W = [[T1/T1, T1/T2, T1/T3, T1/T4, T1/T5, T1/T6, T1/T7, T1/T8, T1/T9, T1/T10],
         [T2/T1, T2/T2, T2/T3, T2/T4, T2/T5, T2/T6, T2/T7, T2/T8, T2/T9, T2/T10],
         [T3/T1, T3/T2, T3/T3, T3/T4, T3/T5, T3/T6, T3/T7, T3/T8, T3/T9, T3/T10],
         [T4/T1, T4/T2, T4/T3, T4/T4, T4/T5, T4/T6, T4/T7, T4/T8, T4/T9, T4/T10],
         [T5/T1, T5/T2, T5/T3, T5/T4, T5/T5, T5/T6, T5/T7, T5/T8, T5/T9, T5/T10],
         [T6/T1, T6/T2, T6/T3, T6/T4, T6/T5, T6/T6, T6/T7, T6/T8, T6/T9, T6/T10],
         [T7/T1, T7/T2, T7/T3, T7/T4, T7/T5, T7/T6, T7/T7, T7/T8, T7/T9, T7/T10],
         [T8/T1, T8/T2, T8/T3, T8/T4, T8/T5, T8/T6, T8/T7, T8/T8, T8/T9, T8/T10],
         [T9/T1, T9/T2, T9/T3, T9/T4, T9/T5, T9/T6, T9/T7, T9/T8, T9/T9, T9/T10],
         [T10/T1, T10/T2, T10/T3, T10/T4, T10/T5, T10/T6, T10/T7, T10/T8, T10/T9, T10/T10]]

T_W_a = np.array(T_W)

                ## Part A: Find Criteria Weights and Consistancy##

# Step 1: Sum the columns
# sum of columns for Criteria Weight One Matrix


def column_sum(T_W):
    return[sum(i) for i in zip(*T_W)]
    global T_W_col_sum


T_W_col_sum = np.array(column_sum(T_W))

# Step 2: Normalize the matrix
T_W_N = T_W_a / T_W_col_sum

# Step 3: Calculate Criteria Weights by finding the row averages
C_T_W =[sum(i) for i in T_W_N]

# convert to an array
# adding brackets to make 2-D array
C_T_W_a = np.array([C_T_W])

# number of indicators
n = 10

# Find the average
Avg_C_T_W = C_T_W_a / n

# Step 4 Find the Consistency ratio
# Step 4a: Calculate the weighted matrix by multiplying the matrix by the criteria weight
WM_T = np.matmul(T_W_a.T, C_T_W_a.T)

# Step 4b: Sum the rows of the weighted matrix to find the weighted sum value
WS_T =[sum(i) for i in WM_T]

# convert to an array
WS_T_a = np.array(WS_T)


# Step 4c: divide the weighted sum value by the criteria weights
R_T = WS_T_a / C_T_W

# Step 4d: Find the Consistency index by calculating (delta max - n)/(n-1)
delta_maxT = (sum(R_T))/n

CI_T = (delta_maxT - n) / (n - 1)


# Step 4e: Divide the Consistency index by the Random index

                            # RI Values
# n=5, RI=1.12; n=6, RI=1.24; n=7, RI=1.32; n=8, RI=1.41; n=9, RI=1.45; n=10, RI=1.49; n=11, RI=1.51; n=12, RI=1.54
# If CR < 0.1 then our matrix is consistent
RI = 1.49
CR_T = CI_T / RI

# Step 1: Assign criteria weights in matrix

Env_W = [[Env1/Env1, Env1/Env2, Env1/Env3, Env1/Env4, Env1/Env5, Env1/Env6],
         [Env2/Env1, Env2/Env2, Env2/Env3, Env2/Env4, Env2/Env5, Env2/Env6],
         [Env3/Env1, Env3/Env2, Env3/Env3, Env3/Env4, Env3/Env5, Env3/Env6],
         [Env4/Env1, Env4/Env2, Env4/Env3, Env4/Env4, Env4/Env5, Env4/Env6],
         [Env5/Env1, Env5/Env2, Env5/Env3, Env5/Env4, Env5/Env5, Env5/Env6],
         [Env6/Env1, Env6/Env2, Env6/Env3, Env6/Env4, Env6/Env5, Env6/Env6]]

Env_W_a = np.array(Env_W)

                ## Part A: Find Criteria Weights and Consistancy##
# Step 1: Sum the columns
# sum of columns for Criteria Weight One Matrix


def column_sum(Env_W):
    return[sum(i) for i in zip(*Env_W)]
    global Env_W_col_sum


Env_W_col_sum = np.array(column_sum(Env_W))


# Step 2: Normalize the matrix
Env_W_N = Env_W_a / Env_W_col_sum


# Step 3: Calculate Criteria Weights by finding the row averages
C_Env_W =[sum(i) for i in Env_W_N]

# convert to an array
# adding brackets to make 2-D array
C_Env_W_a = np.array([C_Env_W])

# #### have a function that counts the columns and inputs the value as n number of indicators
n = 6

# Find the average
Avg_C_Env_W = C_Env_W_a / n

# Step 4 Find the Consistency ratio
# Step 4a: Calculate the weighted matrix by multiplying the matrix by the criteria weight
WM_Env = np.matmul(Env_W_a.Env, C_Env_W_a.Env)

# Step 4b: Sum the rows of the weighted matrix to find the weighted sum value
WS_Env =[sum(i) for i in WM_Env]

# convert to an array
WS_Env_a = np.array(WS_Env)


# Step 4c: divide the weighted sum value by the criteria weights
R_Env = WS_Env_a / C_Env_W

# Step 4d: Find the Consistency index by calculating (delta max - n)/(n-1)
delta_maxEnv = (sum(R_Env))/n

CI_Env = (delta_maxEnv - n) / (n - 1)


# Step 4e: Divide the Consistency index by the Random index

                            # RI Values
# n=5, RI=1.12; n=6, RI=1.24; n=7, RI=1.32; n=8, RI=1.41; n=9, RI=1.45; n=10, RI=1.49; n=11, RI=1.51; n=12, RI=1.54
# If CR < 0.1 then our matrix is consistent
RI = 1.24
CR_Env = CI_Env / RI

# Step 1: Assign criteria weights in matrix

LCA_W = [[LCA1/LCA1, LCA1/LCA2, LCA1/LCA3, LCA1/LCA4, LCA1/LCA5, LCA1/LCA6],
          [LCA2/LCA1, LCA2/LCA2, LCA2/LCA3, LCA2/LCA4, LCA2/LCA5, LCA2/LCA6],
          [LCA3/LCA1, LCA3/LCA2, LCA3/LCA3, LCA3/LCA4, LCA3/LCA5, LCA3/LCA6],
          [LCA4/LCA1, LCA4/LCA2, LCA4/LCA3, LCA4/LCA4, LCA4/LCA5, LCA4/LCA6],
          [LCA5/LCA1, LCA5/LCA2, LCA5/LCA3, LCA5/LCA4, LCA5/LCA5, LCA5/LCA6],
          [LCA6/LCA1, LCA6/LCA2, LCA6/LCA3, LCA6/LCA4, LCA6/LCA5, LCA6/LCA6]]

LCA_W_a = np.array(LCA_W)

                ## Part A: Find Criteria Weights and Consistancy##
# Step 1: Sum the columns
# sum of columns for Criteria Weight One Matrix


def column_sum(LCA_W):
    return[sum(i) for i in zip(*LCA_W)]
    global LCA_W_col_sum


LCA_W_col_sum = np.array(column_sum(LCA_W))


# Step 2: Normalize the matrix
LCA_W_N = LCA_W_a / LCA_W_col_sum


# Step 3: Calculate Criteria Weights by finding the row averages
C_LCA_W =[sum(i) for i in LCA_W_N]

# convert to an array
# adding brackets to make 2-D array
C_LCA_W_a = np.array([C_LCA_W])

# #### have a function that counts the columns and inputs the value as n number of indicators
n = 6

# Find the average
Avg_C_LCA_W = C_LCA_W_a / n

# Step 4 Find the Consistency ratio
# Step 4a: Calculate the weighted matrix by multiplying the matrix by the criteria weight
WM_LCA = np.matmul(LCA_W_a.LCA, C_LCA_W_a.LCA)

# Step 4b: Sum the rows of the weighted matrix to find the weighted sum value
WS_LCA =[sum(i) for i in WM_LCA]

# convert to an array
WS_LCA_a = np.array(WS_LCA)


# Step 4c: divide the weighted sum value by the criteria weights
R_LCA = WS_LCA_a / C_LCA_W

# Step 4d: Find the Consistency index by calculating (delta max - n)/(n-1)
delta_maxLCA = (sum(R_LCA))/n

CI_LCA = (delta_maxLCA - n) / (n - 1)


# Step 4e: Divide the Consistency index by the Random index

                            # RI Values
# n=5, RI=1.12; n=6, RI=1.24; n=7, RI=1.32; n=8, RI=1.41; n=9, RI=1.45; n=10, RI=1.49; n=11, RI=1.51; n=12, RI=1.54
# If CR < 0.1 then our matrix is consistent
RI = 1.24
CR_LCA = CI_LCA / RI

# Step 1: Assign criteria weights in matrix

S_W = [[S1/S1, S1/S2, S1/S3, S1/S4, S1/S5, S1/S6, S1/S7, S1/S8, S1/S9, S1/S10, S1/S11, S1/S12],
         [S2/S1, S2/S2, S2/S3, S2/S4, S2/S5, S2/S6, S2/S7, S2/S8, S2/S9, S2/S10, S2/S11, S2/S12],
         [S3/S1, S3/S2, S3/S3, S3/S4, S3/S5, S3/S6, S3/S7, S3/S8, S3/S9, S3/S10, S3/S11, S3/S12],
         [S4/S1, S4/S2, S4/S3, S4/S4, S4/S5, S4/S6, S4/S7, S4/S8, S4/S9, S4/S10, S4/S11, S4/S12],
         [S5/S1, S5/S2, S5/S3, S5/S4, S5/S5, S5/S6, S5/S7, S5/S8, S5/S9, S5/S10, S5/S11, S5/S12],
         [S6/S1, S6/S2, S6/S3, S6/S4, S6/S5, S6/S6, S6/S7, S6/S8, S6/S9, S6/S10, S6/S11, S6/S12],
         [S7/S1, S7/S2, S7/S3, S7/S4, S7/S5, S7/S6, S7/S7, S7/S8, S7/S9, S7/S10, S7/S11, S7/S12],
         [S8/S1, S8/S2, S8/S3, S8/S4, S8/S5, S8/S6, S8/S7, S8/S8, S8/S9, S8/S10, S8/S11, S8/S12],
         [S9/S1, S9/S2, S9/S3, S9/S4, S9/S5, S9/S6, S9/S7, S9/S8, S9/S9, S9/S10, S9/S11, S9/S12],
         [S10/S1, S10/S2, S10/S3, S10/S4, S10/S5, S10/S6, S10/S7, S10/S8, S10/S9, S10/S10, S10/S11, S10/S12],
         [S11/S1, S11/S2, S11/S3, S11/S4, S11/S5, S11/S6, S11/S7, S11/S8, S11/S9, S11/S10, S11/S11, S11/S12],
         [S12/S1, S12/S2, S12/S3, S12/S4, S12/S5, S12/S6, S12/S7, S12/S8, S12/S9, S12/S10, S12/S11, S12/S12]]

S_W_a = np.array(S_W)

                ## Part A: Find Criteria Weights and Consistency ##
# Step 1: Sum the columns
# sum of columns for Criteria Weight One Matrix


def column_sum(S_W):
    return[sum(i) for i in zip(*S_W)]
    global S_W_col_sum


S_W_col_sum = np.array(column_sum(S_W))


# Step 2: Normalize She matrix
S_W_N = S_W_a / S_W_col_sum


# Step 3: Calculate Criteria Weights by finding the row averages
C_S_W =[sum(i) for i in S_W_N]

# convert to an array
# adding brackets to make 2-D array
C_S_W_a = np.array([C_S_W])

# number of indicators
n = 10

# Find the average
Avg_C_S_W = C_S_W_a / n

# Step 4 Find the Consistency ratio
# Step 4a: Calculate the weighted matrix by multiplying the matrix by the criteria weight
WM_S = np.maSmul(S_W_a.S, C_S_W_a.S)

# Step 4b: Sum the rows of the weighted matrix to find the weighted sum value
WS_S =[sum(i) for i in WM_S]

# convert to an array
WS_S_a = np.array(WS_S)


# Step 4c: divide the weighted sum value by the criteria weights
R_S = WS_S_a / C_S_W

# Step 4d: Find the Consistency index by calculating (delta max - n)/(n-1)
delta_maxS = (sum(R_S))/n

CI_S = (delta_maxS - n) / (n - 1)


# Step 4e: Divide the Consistency index by the Random index

                            # RI Values
# n=5, RI=1.12; n=6, RI=1.24; n=7, RI=1.32; n=8, RI=1.41; n=9, RI=1.45; n=10, RI=1.49; n=11, RI=1.51; n=12, RI=1.54
# If CR < 0.1 then our matrix is consistent
RI = 1.54
CR_S = CI_S / RI