# -*- coding: utf-8 -*-
"""
Modified on Fri July 2, 2021

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

# ## DMsan Model ##
# from __future__ import division #floating point division in Py2x
import numpy as np
# import math
import pandas as pd
import country_converter as coco
import os
from scipy.stats import rankdata

# Data files associated with the model
data_path = os.path.join(os.path.dirname(__file__), 'data')
result_path = os.path.join(os.path.dirname(__file__), 'results')
data_path_tech_scores = os.path.join(data_path, 'technology_scores.xlsx')
data_path_weight_scenarios = os.path.join(data_path, 'criteria_weight_scenarios.xlsx')

# Step 1: Identify Location
# !!! Input location by country in the single quotations in the brackets
location = ['Uganda']

# converting location to match the database
location = coco.convert(names=location, to='name_short')


# number_of_alternatives
# !!! input the number of alternatives evaluated in the model,
# this should be an integer
m = 3

# default values when an indicator is N/A
X = 0.00001

# Local Weights #

# =============================================================================
# Criteria: Social
# =============================================================================
# Sub-criteria: End-user acceptability
# !!! Input community preference
# Local Weight Indicator S3: Disposal convenience preference for user
# relates to the preference for disposal requirements on the user end
# if management is responsible for disposal or if the community did not describe
# this indicator as important, then insert X
# 0 being low preference to frequency of disposal to 100 being high preference for frequency of disposal
# ## Specific to Bwaise example: community did not mention disposal as affecting their acceptability ##
S3 = X

# !!! Input community preference
# Local Weight Indicator S4: Cleaning preference
# relates to the preference for cleaning requirements
# 0 being low preference to frequency of cleaning to 100 being high preference for frequency of cleaning
S4 = 44

# !!! Input community preference
# Local Weight Indicator S5: Privacy preference
# relates to the preference for privacy (# of households sharing a system)
# 0 being low preference for privacy to 100 being high preference for privacy
S5 = 47

# !!! Input community preference
# Local Weight Indicator S6: Odor preference
# relates to the preference of odor with
# 0 being low preference for odor to 100 being high preference for odor
S6 = 22

# !!! Input community preference
# Local Weight Indicator S7: Security preference
# relates to the preference of security with
# 0 being low preference for secutiy to 100 being high preference for odor
# ## Specific to Bwaise example: community did not mention disposal as affecting their acceptability ##
S7 = X

# Sub-criteria: Management Acceptability
# !!! Input management (i.e., landlord) preference
# Local Weight Indicator S8: Disposal convenience preference
# relates to the preference for disposal requirements
# 0 being low importance to frequency of disposal to 100 being high importance for frequency of disposal
# ## Specific to Bwaise example: the sanitation system is controlled by the end-usr, not landlord ##
S8 = X

# ## Input management preference ##
# Local Weight Indicator S9: Cleaning preference
# relates to the preference for cleaning requirements
# 0 being low importance to frequency of cleaning to 100 being high importance for frequency of cleaning
S9 = X

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

# Sub-criteria: End-user acceptability
# S3-S7 were manually inputed above

# Sub-criteria: Management acceptability
# S8-S9 were manually inputed above


# =============================================================================
# Criteria: Technical
# =============================================================================
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
# relates to the grid-electricity flexibility
electricity_blackouts = pd.read_excel(data_path+'/location.xlsx', sheet_name='ElectricityBlackouts',
                                      index_col='Country')
t8 = (electricity_blackouts.loc[location, 'Value'])
T8 = (100-(t8/72.5*100))

# Local Weight Indicator T9:
# relates to the drought flexibility
water_stress = pd.read_excel(data_path+'/location.xlsx', sheet_name='WaterStress', index_col='Country')
t9 = (water_stress.loc[location, 'Value'])
T9 = (100-(t9/4.82*100))


# =============================================================================
# Criteria: Resource Recovery Potential
# =============================================================================
# Local Weight Indicator RR1:
# relates to the water stress (Water Recovery)
RR1 = T9

# Local Weight Indicator RR2:
# relates to nitrogen (N) fertilizer fulfillment (Nutrient Recovery)
n_fertilizer_fulfillment = pd.read_excel(data_path+'/location.xlsx', sheet_name='NFertilizerFulfillment',
                                         index_col='Country')
rr2 = (n_fertilizer_fulfillment.loc[location, 'Value'])
RR2 = (1 - (rr2/100)) * 100

# Local Weight Indicator RR3:
# relates to phosphorus (P) fertilizer fulfillment (Nutrient Recovery)
p_fertilizer_fulfillment = pd.read_excel(data_path+'/location.xlsx', sheet_name='PFertilizerFulfillment',
                                         index_col='Country')
rr3 = (p_fertilizer_fulfillment.loc[location, 'Value'])
RR3 = (1 - (rr3/100)) * 100

# Local Weight Indicator RR4:
# relates to potassium (K) fertilizer fulfillment (Nutrient Recovery)
k_fertilizer_fulfillment = pd.read_excel(data_path+'/location.xlsx', sheet_name='KFertilizerFulfillment',
                                         index_col='Country')
rr4 = (k_fertilizer_fulfillment.loc[location, 'Value'])
RR4 = (1 - (rr4/100)) * 100

# Local Weight Indicator RR5:
# relates to renewable energy consumption (Energy Recovery)
renewable_energy = pd.read_excel(data_path+'/location.xlsx', sheet_name='RenewableEnergyConsumption',
                                 index_col='Country')
rr5 = (renewable_energy.loc[location, 'Value'])
RR5 = (1 - (rr5/100)) * 100

# Local Weight Indicator RR6:
# relates to infrastructure quality (Supply Chain Infrastructure)
infrastructure = pd.read_excel(data_path+'/location.xlsx', sheet_name='InfrastructureQuality',
                                         index_col='Country')
rr6 = (infrastructure.loc[location, 'Value'])
RR6 = (1 - (rr6/7)) * 100


# =============================================================================
# Criteria: Environmental
# =============================================================================
# Local Weight Indicator Env1:
# relates to the ecosystem quality (LCA)
Env1 = 0.34

# Local Weight Indicator Env2:
# relates to the human health (LCA)
Env2 = 0.33

# Local Weight Indicator Env3:
# relates to the resource depletion (LCA)
Env3 = 0.33


# Tech/System Performance Scores #

# Technology Performance Values - Technical Criteria
Tech_Score_T1 = pd.read_excel(data_path_tech_scores, sheet_name='user_interface').expected
Tech_Score_T2 = pd.read_excel(data_path_tech_scores, sheet_name='treatment_type').expected
Tech_Score_T3 = pd.read_excel(data_path_tech_scores, sheet_name='system_part_accessibility').expected
Tech_Score_T4 = pd.read_excel(data_path_tech_scores, sheet_name='design_transport').expected
Tech_Score_T5 = pd.read_excel(data_path_tech_scores, sheet_name='construction_skills').expected
Tech_Score_T6 = pd.read_excel(data_path_tech_scores, sheet_name='OM_complexity').expected
Tech_Score_T7 = pd.read_excel(data_path_tech_scores, sheet_name='pop_flexibility').expected
Tech_Score_T8 = pd.read_excel(data_path_tech_scores, sheet_name='electricity_flexibility').expected
Tech_Score_T9 = pd.read_excel(data_path_tech_scores, sheet_name='drought_flexibility').expected

Tech_Score_T_All = pd.DataFrame([Tech_Score_T1, Tech_Score_T2, Tech_Score_T3, Tech_Score_T4,
                                 Tech_Score_T5, Tech_Score_T6, Tech_Score_T7, Tech_Score_T8,
                                 Tech_Score_T9]).transpose()
Tech_Score_T_All.columns = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']

# Technology Performance Values - Resource Recovery Criteria
Tech_Score_RR1 = pd.read_excel(data_path_tech_scores, sheet_name='water_reuse').expected

baseline = pd.read_csv(os.path.join(data_path, 'bwaise_baseline.tsv'), index_col=(0, 1), sep='\t')
Tech_Score_RR2 = baseline.loc[('Net recovery', 'N')].values
Tech_Score_RR3 = baseline.loc[('Net recovery', 'P')].values
Tech_Score_RR4 = baseline.loc[('Net recovery', 'K')].values
Tech_Score_RR5 = baseline.loc[('Net recovery', 'energy')].values
Tech_Score_RR6 = pd.read_excel(data_path_tech_scores, sheet_name='supply_chain').expected

Tech_Score_RR_All = pd.DataFrame([Tech_Score_RR1, Tech_Score_RR2, Tech_Score_RR3, Tech_Score_RR4, Tech_Score_RR5,
                                  Tech_Score_RR6]).transpose()
Tech_Score_RR_All.columns = ['RR1', 'RR2', 'RR3', 'RR4', 'RR5', 'RR6']

# Technology Performance Values - Environmental (LCA) Criteria
H_ind = [ind for ind in baseline.index if ind[1].startswith('H_')]
Tech_Score_Env1 = baseline[baseline.index == H_ind[0]].values[0]  # Ecosystem Quality
Tech_Score_Env2 = baseline[baseline.index == H_ind[1]].values[0]  # Human Health
Tech_Score_Env3 = baseline[baseline.index == H_ind[2]].values[0]  # Resource Depletion

Tech_Score_Env_All = pd.DataFrame([Tech_Score_Env1, Tech_Score_Env2, Tech_Score_Env3]).transpose()
Tech_Score_Env_All.columns = ['Env1', 'Env2', 'Env3']

# Technology Performance Values - Economic Criteria
Tech_Score_Econ1 = baseline.loc[('TEA results', 'Net cost')].values

Tech_Score_Econ_All = pd.DataFrame([Tech_Score_Econ1]).transpose()
Tech_Score_Econ_All.columns = ['Econ1']

# Technology Performance Values - Social/Institutional Criteria
Tech_Score_S1 = pd.read_excel(data_path_tech_scores, sheet_name='design_job_creation').expected
Tech_Score_S2 = pd.read_excel(data_path_tech_scores, sheet_name='design_high_pay_jobs').expected
Tech_Score_S3 = pd.read_excel(data_path_tech_scores, sheet_name='end_user_disposal').expected
Tech_Score_S4 = pd.read_excel(data_path_tech_scores, sheet_name='end_user_cleaning').expected
Tech_Score_S5 = pd.read_excel(data_path_tech_scores, sheet_name='privacy').expected
Tech_Score_S6 = pd.read_excel(data_path_tech_scores, sheet_name='odor').expected
Tech_Score_S7 = pd.read_excel(data_path_tech_scores, sheet_name='security').expected
Tech_Score_S8 = pd.read_excel(data_path_tech_scores, sheet_name='management_disposal').expected
Tech_Score_S9 = pd.read_excel(data_path_tech_scores, sheet_name='management_cleaning').expected

Tech_Score_S_All = pd.DataFrame([Tech_Score_S1, Tech_Score_S2, Tech_Score_S3, Tech_Score_S4, Tech_Score_S5,
                                 Tech_Score_S6, Tech_Score_S7, Tech_Score_S8, Tech_Score_S9]).transpose()
Tech_Score_S_All.columns = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']

# Technology Performance Values - Compiled Criteria
Tech_Scores_compiled = pd.concat([Tech_Score_T_All, Tech_Score_RR_All, Tech_Score_Env_All, Tech_Score_Econ_All,
                                  Tech_Score_S_All], axis=1)


# AHP to Determine Sub-Criteria Weights #

# Technological/Functional Sub-criteria Matrix
# Step 1: Assign criteria weights in matrix
T_W = [[T1/T1, T1/T2, T1/T3, T1/T4, T1/T5, T1/T6, T1/T7, T1/T8, T1/T9],
       [T2/T1, T2/T2, T2/T3, T2/T4, T2/T5, T2/T6, T2/T7, T2/T8, T2/T9],
       [T3/T1, T3/T2, T3/T3, T3/T4, T3/T5, T3/T6, T3/T7, T3/T8, T3/T9],
       [T4/T1, T4/T2, T4/T3, T4/T4, T4/T5, T4/T6, T4/T7, T4/T8, T4/T9],
       [T5/T1, T5/T2, T5/T3, T5/T4, T5/T5, T5/T6, T5/T7, T5/T8, T5/T9],
       [T6/T1, T6/T2, T6/T3, T6/T4, T6/T5, T6/T6, T6/T7, T6/T8, T6/T9],
       [T7/T1, T7/T2, T7/T3, T7/T4, T7/T5, T7/T6, T7/T7, T7/T8, T7/T9],
       [T8/T1, T8/T2, T8/T3, T8/T4, T8/T5, T8/T6, T8/T7, T8/T8, T8/T9],
       [T9/T1, T9/T2, T9/T3, T9/T4, T9/T5, T9/T6, T9/T7, T9/T8, T9/T9]]

T_W_a = np.array(T_W)

# Step 2: Sum the columns
# sum of columns for Criteria Weight One Matrix
def column_sum(T_W):
    return[sum(i) for i in zip(*T_W)]
    global T_W_col_sum
T_W_col_sum = np.array(column_sum(T_W))

# Step 3: Normalize the matrix
T_W_N = T_W_a / T_W_col_sum

# Step 4: Calculate Criteria Weights by finding the row averages
C_T_W =[sum(i) for i in T_W_N]

# convert to an array
# adding brackets to make 2-D array
C_T_W_a = np.array([C_T_W])

# number of indicators
n = 9

# Find the average
Avg_C_T_W = C_T_W_a / n

# Step 5 Find the Consistency ratio
# Step 5a: Calculate the weighted matrix by multiplying the matrix by the criteria weight
WM_T = np.matmul(T_W_a.T, C_T_W_a.T)

# Step 5b: Sum the rows of the weighted matrix to find the weighted sum value
WS_T = [sum(i) for i in WM_T]

# convert to an array
WS_T_a = np.array(WS_T)

# Step 5c: divide the weighted sum value by the criteria weights
R_T = WS_T_a / C_T_W

# Step 5d: Find the Consistency index by calculating (delta max - n)/(n-1)
delta_maxT = (sum(R_T))/n
CI_T = (delta_maxT - n) / (n - 1)

# Step 5e: Divide the Consistency index by the Random index
# RI Values
# n=3, RI=0.58; n=4, RI=0.90; n=5, RI=1.12; n=6, RI=1.24; n=7, RI=1.32; n=8, RI=1.41; n=9, RI=1.45; n=10, RI=1.49;
# n=11, RI=1.51; n=12, RI=1.54
# If CR < 0.1 then our matrix is consistent
RI = 1.45
CR_T = CI_T / RI

# Resource Recovery Sub-Criteria Matrix
# Step 1: Assign criteria weights in matrix
RR_W = [[RR1/RR1, RR1/RR2, RR1/RR3, RR1/RR4, RR1/RR5, RR1/RR6],
         [RR2/RR1, RR2/RR2, RR2/RR3, RR2/RR4, RR2/RR5, RR2/RR6],
         [RR3/RR1, RR3/RR2, RR3/RR3, RR3/RR4, RR3/RR5, RR3/RR6],
         [RR4/RR1, RR4/RR2, RR4/RR3, RR4/RR4, RR4/RR5, RR4/RR6],
         [RR5/RR1, RR5/RR2, RR5/RR3, RR5/RR4, RR5/RR5, RR5/RR6],
         [RR6/RR1, RR6/RR2, RR6/RR3, RR6/RR4, RR6/RR5, RR6/RR6]]
RR_W_a = np.array(RR_W)

# Step 2: Sum the columns
# sum of columns for Criteria Weight One Matrix
def column_sum(RR_W):
    return[sum(i) for i in zip(*RR_W)]
    global RR_W_col_sum
RR_W_col_sum = np.array(column_sum(RR_W))

# Step 3: Normalize the matrix
RR_W_N = RR_W_a / RR_W_col_sum

# Step 4: Calculate Criteria Weights by finding the row averages
C_RR_W =[sum(i) for i in RR_W_N]

# convert to an array
# adding brackets to make 2-D array
C_RR_W_a = np.array([C_RR_W])

# #### have a function that counts the columns and inputs the value as n number of indicators
n = 6

# Find the average
Avg_C_RR_W = C_RR_W_a / n

# Step 5 Find the Consistency ratio
# Step 5a: Calculate the weighted matrix by multiplying the matrix by the criteria weight
WM_RR = np.matmul(RR_W_a.T, C_RR_W_a.T)

# Step 5b: Sum the rows of the weighted matrix to find the weighted sum value
WS_RR = [sum(i) for i in WM_RR]

# convert to an array
WS_RR_a = np.array(WS_RR)

# Step 5c: divide the weighted sum value by the criteria weights
R_RR = WS_RR_a / C_RR_W

# Step 5d: Find the Consistency index by calculating (delta max - n)/(n-1)
delta_maxRR = (sum(R_RR))/n
CI_RR = (delta_maxRR - n) / (n - 1)

# Step 5e: Divide the Consistency index by the Random index
# RI Values
# n=3, R.I.=0.58; n=4, R.I.=0.90; n=5, RI=1.12; n=6, RI=1.24; n=7, RI=1.32; n=8, RI=1.41; n=9, RI=1.45; n=10, RI=1.49;
# n=11, RI=1.51; n=12, RI=1.54
# If CR < 0.1 then our matrix is consistent
RI = 1.24
CR_RR = CI_RR / RI

# Environmental Sub-Criteria Matrix
# Step 1: Assign criteria weights in matrix
Env_W = [[Env1/Env1, Env1/Env2, Env1/Env3],
          [Env2/Env1, Env2/Env2, Env2/Env3],
          [Env3/Env1, Env3/Env2, Env3/Env3]]

Env_W_a = np.array(Env_W)

# Step 2: Sum the columns
# sum of columns for Criteria Weight One Matrix
def column_sum(Env_W):
    return[sum(i) for i in zip(*Env_W)]
    global Env_W_col_sum
Env_W_col_sum = np.array(column_sum(Env_W))

# Step 3: Normalize the matrix
Env_W_N = Env_W_a / Env_W_col_sum

# Step 4: Calculate Criteria Weights by finding the row averages
C_Env_W = [sum(i) for i in Env_W_N]

# convert to an array
# adding brackets to make 2-D array
C_Env_W_a = np.array([C_Env_W])

# #### have a function that counts the columns and inputs the value as n number of indicators
n = 3

# Find the average
Avg_C_Env_W = C_Env_W_a / n

# Step 5 Find the Consistency ratio
# Step 5a: Calculate the weighted matrix by multiplying the matrix by the criteria weight
WM_Env = np.matmul(Env_W_a.T, C_Env_W_a.T)

# Step 5b: Sum the rows of the weighted matrix to find the weighted sum value
WS_Env =[sum(i) for i in WM_Env]

# convert to an array
WS_Env_a = np.array(WS_Env)

# Step 5c: divide the weighted sum value by the criteria weights
R_Env = WS_Env_a / C_Env_W

# Step 5d: Find the Consistency index by calculating (delta max - n)/(n-1)
delta_maxEnv = (sum(R_Env))/n

CI_Env = (delta_maxEnv - n) / (n - 1)

# Step 5e: Divide the Consistency index by the Random index
# RI Values
# n=3, R.I.=0.58; n=4, R.I.=0.90; n=5, RI=1.12; n=6, RI=1.24; n=7, RI=1.32; n=8, RI=1.41; n=9, RI=1.45; n=10, RI=1.49;
# n=11, RI=1.51; n=12, RI=1.54
# If CR < 0.1 then our matrix is consistent
RI = 0.58
CR_Env = CI_Env / RI

# ## Econ is not evluated as it has only one indicator value

# Social Sub-Criteria Matrix
# Step 1: Assign criteria weights in matrix
S_W = [[S1/S1, S1/S2, S1/S3, S1/S4, S1/S5, S1/S6, S1/S7, S1/S8, S1/S9],
         [S2/S1, S2/S2, S2/S3, S2/S4, S2/S5, S2/S6, S2/S7, S2/S8, S2/S9],
         [S3/S1, S3/S2, S3/S3, S3/S4, S3/S5, S3/S6, S3/S7, S3/S8, S3/S9],
         [S4/S1, S4/S2, S4/S3, S4/S4, S4/S5, S4/S6, S4/S7, S4/S8, S4/S9],
         [S5/S1, S5/S2, S5/S3, S5/S4, S5/S5, S5/S6, S5/S7, S5/S8, S5/S9],
         [S6/S1, S6/S2, S6/S3, S6/S4, S6/S5, S6/S6, S6/S7, S6/S8, S6/S9],
         [S7/S1, S7/S2, S7/S3, S7/S4, S7/S5, S7/S6, S7/S7, S7/S8, S7/S9],
         [S8/S1, S8/S2, S8/S3, S8/S4, S8/S5, S8/S6, S8/S7, S8/S8, S8/S9],
         [S9/S1, S9/S2, S9/S3, S9/S4, S9/S5, S9/S6, S9/S7, S9/S8, S9/S9]]
S_W_a = np.array(S_W)

# Step 2: Sum the columns
# sum of columns for Criteria Weight One Matrix
def column_sum(S_W):
    return[sum(i) for i in zip(*S_W)]
    global S_W_col_sum
S_W_col_sum = np.array(column_sum(S_W))

# Step 3: Normalize the matrix
S_W_N = S_W_a / S_W_col_sum

# Step 4: Calculate Criteria Weights by finding the row averages
C_S_W =[sum(i) for i in S_W_N]

# convert to an array
# adding brackets to make 2-D array
C_S_W_a = np.array([C_S_W])

# number of indicators
n = 9

# Find the average
Avg_C_S_W = C_S_W_a / n

# Step 5 Find the Consistency ratio
# Step 5a: Calculate the weighted matrix by multiplying the matrix by the criteria weight
WM_S = np.matmul(S_W_a.T, C_S_W_a.T)

# Step 5b: Sum the rows of the weighted matrix to find the weighted sum value
WS_S =[sum(i) for i in WM_S]

# convert to an array
WS_S_a = np.array(WS_S)

# Step 5c: divide the weighted sum value by the criteria weights
R_S = WS_S_a / C_S_W

# Step 5d: Find the Consistency index by calculating (delta max - n)/(n-1)
delta_maxS = (sum(R_S))/n
CI_S = (delta_maxS - n) / (n - 1)

# Step 5e: Divide the Consistency index by the Random index
# RI Values
# n=3, R.I.=0.58; n=4, R.I.=0.90; n=5, RI=1.12; n=6, RI=1.24; n=7, RI=1.32; n=8, RI=1.41; n=9, RI=1.45; n=10, RI=1.49;
# n=11, RI=1.51; n=12, RI=1.54
# If CR < 0.1 then our matrix is consistent
RI = 1.45
CR_S = CI_S / RI


# TOPSIS Methodology to Rank Alternatives #

#  Create data frames of sub-criteria weights

# Data frame of technical sub-criteria weights
T_subcriteria_weights = pd.DataFrame(Avg_C_T_W)
T_subcriteria_weights.columns = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9']

# Data frame of resource recovery sub-criteria weights
RR_subcriteria_weights = pd.DataFrame(Avg_C_RR_W)
RR_subcriteria_weights.columns = ['RR1', 'RR2', 'RR3', 'RR4', 'RR5', 'RR6']

# Data frame of environmental (LCA) sub-criteria weights
Env_subcriteria_weights = pd.DataFrame(Avg_C_Env_W)
Env_subcriteria_weights.columns = ['Env1', 'Env2', 'Env3']

# Data frame of economic sub-criteria weights
Econ_subcriteria_weights = pd.DataFrame([1])
Econ_subcriteria_weights.columns = ['Econ1']

# Data frame of social/institutional sub-criteria weights
S_subcriteria_weights = pd.DataFrame(Avg_C_S_W)
S_subcriteria_weights.columns = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']

# Compiled sub-criteria weights
subcriteria_weights = pd.concat([T_subcriteria_weights, RR_subcriteria_weights, Env_subcriteria_weights,
                                 Econ_subcriteria_weights, S_subcriteria_weights], axis=1)

# Data to perform TOPSIS
perform_values = Tech_Scores_compiled
criteria_weight = pd.read_excel(data_path+'/criteria_weight_scenarios.xlsx', sheet_name='weight_scenarios')
indicator_weight = subcriteria_weights
indicator_type = pd.read_excel(data_path+'/criteria_weight_scenarios.xlsx', sheet_name='indicator_type')

num_weight = criteria_weight.shape[0]  # quantity of criteria weighting scenarios to run
num_system = perform_values.shape[0]  # quantity of sanitation system alternatives to evaluate
num_indicator = indicator_type.shape[1]  # quantity of indicators included in the model

# Output Excel File of Results
writer1 = pd.ExcelWriter(os.path.join(result_path, 'RESULTS_AHP_TOPSIS.xlsx'))

# Indicator Column Names
indicators = list(indicator_type.columns)
sanitation_systems = pd.read_excel(data_path_tech_scores, sheet_name='user_interface').system

# Step 1: Normalized Decision Matrix (Vector Normalization)
normal_matrix_FINAL = pd.DataFrame()
for i in range(num_indicator):
    normal_array = pd.DataFrame()
    squares = 0  # initialize the denominator for normalization
    for j in range(num_system):
        performance_value = perform_values.iloc[j, i]
        performance_value = float(performance_value)
        squares = squares + performance_value ** 2
        denominator = squares ** 0.5
    for j in range(num_system):
        if denominator == 0:
            normal_value = 0
        else:
            normal_value = perform_values.iloc[j, i] / denominator
        normal_value = pd.DataFrame([normal_value])
        normal_array = pd.concat([normal_array, normal_value]).reset_index(drop=True)
    normal_matrix_FINAL = pd.concat([normal_matrix_FINAL, normal_array], axis=1).reset_index(drop=True)
normal_matrix_FINAL.columns = indicators

# Step 2: Ranking System Alternatives Under Criteria Weighting Scenarios
performance_score_FINAL = pd.DataFrame()
ranking_FINAL = pd.DataFrame()
for i in range(num_weight):
    # Weighted Normalized Decision Matrix
    weighted_normal_matrix_FINAL = pd.DataFrame()
    for j in range(num_indicator):
        weighted_array = pd.DataFrame()
        indicator_criteria = indicator_type.iloc[0, j]
        if indicator_criteria == 'T':
            weight = criteria_weight.loc[i, 'T'] * indicator_weight.iloc[0, j]
        elif indicator_criteria == 'RR':
            weight = criteria_weight.loc[i, 'RR'] * indicator_weight.iloc[0, j]
        elif indicator_criteria == 'Env':
            weight = criteria_weight.loc[i, 'Env'] * indicator_weight.iloc[0, j]
        elif indicator_criteria == 'Econ':
            weight = criteria_weight.loc[i, 'Econ'] * indicator_weight.iloc[0, j]
        elif indicator_criteria == 'S':
            weight = criteria_weight.loc[i, 'S'] * indicator_weight.iloc[0, j]
        else:
            print("Indicator Type Error")
        for k in range(num_system):
            weighted_normal_value = normal_matrix_FINAL.iloc[k, j] * weight
            weighted_normal_value = pd.DataFrame([weighted_normal_value])
            weighted_array = pd.concat([weighted_array, weighted_normal_value]).reset_index(drop=True)
        weighted_normal_matrix_FINAL = pd.concat([weighted_normal_matrix_FINAL, weighted_array], axis=1).reset_index(drop=True)
    weighted_normal_matrix_FINAL.columns = indicators

    ideal_best_FINAL = pd.DataFrame()
    ideal_worst_FINAL = pd.DataFrame()
    best_array = pd.DataFrame()
    worst_array = pd.DataFrame()
    for j in range(num_indicator):
        # Ideal Best and Ideal Worst Value for Each Sub-Criteria
        indicator_category = indicator_type.iloc[1, j]  # 0 is non-beneficial (want low value) and 1 is beneficial
        if indicator_category == 0:  # sub-criteria is non-beneficial, so ideal best is the lowest value
            ideal_best = min(weighted_normal_matrix_FINAL.iloc[:, j])
            ideal_worst = max(weighted_normal_matrix_FINAL.iloc[:, j])
        elif indicator_category == 1:  # sub-criteria is beneficial, so ideal best is the highest value
            ideal_best = max(weighted_normal_matrix_FINAL.iloc[:, j])
            ideal_worst = min(weighted_normal_matrix_FINAL.iloc[:, j])
        else:
            print("Ideal Best and Worst Error")
        ideal_best = pd.DataFrame([ideal_best])
        best_array = pd.concat([best_array, ideal_best])
        ideal_worst = pd.DataFrame([ideal_worst])
        worst_array = pd.concat([worst_array, ideal_worst])
    ideal_best_FINAL = pd.concat([ideal_best_FINAL, best_array]).reset_index(drop=True)
    ideal_worst_FINAL = pd.concat([ideal_worst_FINAL, worst_array]).reset_index(drop=True)

    performance_score_DF = pd.DataFrame()
    ranking_DF = pd.DataFrame()
    for j in range(num_system):
        # Euclidean Distance from Ideal Best and Ideal Worst
        sum_dif_squared_best = 0
        sum_dif_squared_worst = 0
        for k in range(num_indicator):
            dif_squared_best = (weighted_normal_matrix_FINAL.iloc[j, k] - ideal_best_FINAL.iloc[k, :]) ** 2
            dif_squared_worst = (weighted_normal_matrix_FINAL.iloc[j, k] - ideal_worst_FINAL.iloc[k, :]) ** 2
            sum_dif_squared_best = sum_dif_squared_best + dif_squared_best
            sum_dif_squared_worst = sum_dif_squared_worst + dif_squared_worst
        distance_best = sum_dif_squared_best ** 0.5
        distance_worst = sum_dif_squared_worst ** 0.5

        # Performance Score of Each Sanitation System
        performance_score = distance_worst / (distance_best + distance_worst)
        performance_score = pd.DataFrame([performance_score])
        performance_score_DF = pd.concat([performance_score_DF, performance_score], axis=0).reset_index(drop=True)
    breakpoint()
    # Ranking of Each Sanitation System
    rank = (len(performance_score_DF) + 1) - rankdata(performance_score_DF).astype(int)
    ranking_DF = pd.DataFrame(rank).transpose()

    performance_score_DF = performance_score_DF.transpose()
    performance_score_FINAL = pd.concat([performance_score_FINAL, performance_score_DF]).reset_index(drop=True)
    ranking_FINAL = pd.concat([ranking_FINAL, ranking_DF]).reset_index(drop=True)
performance_score_FINAL.columns = sanitation_systems
ranking_FINAL.columns = sanitation_systems

criteria_weight_scenario = pd.read_excel(data_path+'/criteria_weight_scenarios.xlsx', sheet_name='weight_scenarios').Ratio
criteria_weight_scenario = pd.DataFrame(criteria_weight_scenario)
criteria_weight_scenario.columns = ['weight_scenario']

criteria_weight_description = pd.read_excel(data_path+'/criteria_weight_scenarios.xlsx', sheet_name='weight_scenarios').Description
criteria_weight_description = pd.DataFrame(criteria_weight_description)
criteria_weight_description.columns = ['weight_description']

performance_score_FINAL = pd.concat([criteria_weight_scenario, criteria_weight_description, performance_score_FINAL], axis=1)
ranking_FINAL = pd.concat([criteria_weight_scenario, criteria_weight_description, ranking_FINAL], axis=1)

performance_score_FINAL.to_excel(writer1, sheet_name='perform_score')
ranking_FINAL.to_excel(writer1, sheet_name='ranking')

# writer1.save()


# ELECTRE Methodology to Rank Alternatives #

# Output Excel File of Results
writer2 = pd.ExcelWriter(os.path.join(result_path, 'RESULTS_AHP_ELECTRE_-_superior_and_inferior_values.xlsx'))
writer3 = pd.ExcelWriter(os.path.join(result_path, 'RESULTS_AHP_ELECTRE_-_concordance_interval_matrix.xlsx'))
writer4 = pd.ExcelWriter(os.path.join(result_path, 'RESULTS_AHP_ELECTRE_-_concordance_index_matrix.xlsx'))
writer5 = pd.ExcelWriter(os.path.join(result_path, 'RESULTS_AHP_ELECTRE_-_discordance_interval_matrix.xlsx'))
writer6 = pd.ExcelWriter(os.path.join(result_path, 'RESULTS_AHP_ELECTRE_-_discordance_index_matrix.xlsx'))

# Step 1: Forming Decision Making Matrix
# already defined in TOPSIS

# Step 2: Normalize and Weight the Matrix
superior_values_FINAL = pd.DataFrame()
inferior_values_FINAL = pd.DataFrame()

for i2 in range(num_weight):
    # Weighted Normalized Decision Matrix
    weighted_normal_matrix_FINAL = pd.DataFrame()
    for j in range(num_indicator):
        weighted_array = pd.DataFrame()
        indicator_criteria = indicator_type.iloc[0, j]
        if indicator_criteria == 'T':
            weight = criteria_weight.loc[i2, 'T'] * indicator_weight.iloc[0, j]
        elif indicator_criteria == 'RR':
            weight = criteria_weight.loc[i2, 'RR'] * indicator_weight.iloc[0, j]
        elif indicator_criteria == 'Env':
            weight = criteria_weight.loc[i2, 'Env'] * indicator_weight.iloc[0, j]
        elif indicator_criteria == 'Econ':
            weight = criteria_weight.loc[i2, 'Econ'] * indicator_weight.iloc[0, j]
        elif indicator_criteria == 'S':
            weight = criteria_weight.loc[i2, 'S'] * indicator_weight.iloc[0, j]
        else:
            print("Indicator Type Error")
        for k in range(num_system):
            weighted_normal_value = normal_matrix_FINAL.iloc[k, j] * weight
            weighted_normal_value = pd.DataFrame([weighted_normal_value])
            weighted_array = pd.concat([weighted_array, weighted_normal_value]).reset_index(drop=True)
        weighted_normal_matrix_FINAL = pd.concat([weighted_normal_matrix_FINAL, weighted_array], axis=1).reset_index(drop=True)
    weighted_normal_matrix_FINAL.columns = indicators

    # Step 3: Find the concordance and discordance sets
    # loop if A1 is <= to A2 for each indicator for non-beneficial indicators
    # and A1 is >= to A2 for each indicator for beneficial indicators
    concordance = pd.DataFrame()
    for j in range(num_indicator):
        # Ideal Best and Ideal Worst Value for Each Sub-Criteria
        indicator_category = indicator_type.iloc[1, j]  # 0 is non-beneficial (want low value) and 1 is beneficial
        if indicator_category == 0:  # sub-criteria is non-beneficial, so ideal best is the lowest value
            if weighted_normal_matrix_FINAL.iloc[0, j] <= weighted_normal_matrix_FINAL.iloc[1, j]:
                A_1_2 = 1
            else:
                A_1_2 = 0
            if weighted_normal_matrix_FINAL.iloc[0, j] <= weighted_normal_matrix_FINAL.iloc[2, j]:
                A_1_3 = 1
            else:
                A_1_3 = 0
            if weighted_normal_matrix_FINAL.iloc[1, j] <= weighted_normal_matrix_FINAL.iloc[0, j]:
                A_2_1 = 1
            else:
                A_2_1 = 0
            if weighted_normal_matrix_FINAL.iloc[1, j] <= weighted_normal_matrix_FINAL.iloc[2, j]:
                A_2_3 = 1
            else:
                A_2_3 = 0
            if weighted_normal_matrix_FINAL.iloc[2, j] <= weighted_normal_matrix_FINAL.iloc[0, j]:
                A_3_1 = 1
            else:
                A_3_1 = 0
            if weighted_normal_matrix_FINAL.iloc[2, j] <= weighted_normal_matrix_FINAL.iloc[1, j]:
                A_3_2 = 1
            else:
                A_3_2 = 0
        elif indicator_category == 1:  # sub-criteria is beneficial, so ideal best is the highest value
            if weighted_normal_matrix_FINAL.iloc[0, j] >= weighted_normal_matrix_FINAL.iloc[1, j]:
                A_1_2 = 1
            else:
                A_1_2 = 0
            if weighted_normal_matrix_FINAL.iloc[0, j] >= weighted_normal_matrix_FINAL.iloc[2, j]:
                A_1_3 = 1
            else:
                A_1_3 = 0
            if weighted_normal_matrix_FINAL.iloc[1, j] >= weighted_normal_matrix_FINAL.iloc[0, j]:
                A_2_1 = 1
            else:
                A_2_1 = 0
            if weighted_normal_matrix_FINAL.iloc[1, j] >= weighted_normal_matrix_FINAL.iloc[2, j]:
                A_2_3 = 1
            else:
                A_2_3 = 0
            if weighted_normal_matrix_FINAL.iloc[2, j] >= weighted_normal_matrix_FINAL.iloc[0, j]:
                A_3_1 = 1
            else:
                A_3_1 = 0
            if weighted_normal_matrix_FINAL.iloc[2, j] >= weighted_normal_matrix_FINAL.iloc[1, j]:
                A_3_2 = 1
            else:
                A_3_2 = 0
        else:
            print("ELECTRE Input Error")

        # concordance set
        concordance_set = pd.DataFrame([A_1_2, A_1_3, A_2_1, A_2_3, A_3_1, A_3_2])
        concordance = pd.concat([concordance, concordance_set], axis=1)
    concordance.columns = indicators

    # Step 4: Calculate Concordance (Dominance) Interval Matrix
    # will have to fix so it can be a range of alternatives, right now this is for only 3
    # if i = 1 in concordance, pull the sub-criteria weight in same i location
    # find the sum of the is in the row
    concordance_interval_matrix = pd.DataFrame()
    for i in range(concordance_set.shape[0]):
        sum_criteria = 0
        for j in range(num_indicator):
            matrix_value = subcriteria_weights.iloc[0, j]
            if concordance.iloc[i, j] == 1:
                sum_criteria = sum_criteria + matrix_value
            else:
                sum_criteria = sum_criteria
        concordance_interval_matrix_1_1 = 0.0
        concordance_interval_matrix_2_2 = 0.0
        concordance_interval_matrix_3_3 = 0.0
        if i == 0:
            concordance_interval_matrix_1_2 = sum_criteria
        elif i == 1:
            concordance_interval_matrix_1_3 = sum_criteria
        elif i == 2:
            concordance_interval_matrix_2_1 = sum_criteria
        elif i == 3:
            concordance_interval_matrix_2_3 = sum_criteria
        elif i == 4:
            concordance_interval_matrix_3_1 = sum_criteria
        elif i == 5:
            concordance_interval_matrix_3_2 = sum_criteria
    concordance_interval_matrix = pd.DataFrame([[concordance_interval_matrix_1_1, concordance_interval_matrix_1_2, concordance_interval_matrix_1_3],
                                                [concordance_interval_matrix_2_1, concordance_interval_matrix_2_2, concordance_interval_matrix_2_3],
                                                [concordance_interval_matrix_3_1, concordance_interval_matrix_3_2, concordance_interval_matrix_3_3]])

    # Step 5: Calculate Discordance (Weakness) Interval Matrix
    # Calculate absolute value of the difference from the normalized weighted matrix
    # example row 1 - row 2, row 1 - row 3, row 2 - row1, row 2 - row 3, ...
    discordance = pd.DataFrame()
    for row in weighted_normal_matrix_FINAL.iterrows():
        weighted_normal_matrix_FINAL.loc
    df = weighted_normal_matrix_FINAL
    for j in range(num_indicator):
        df.loc['D12'] = abs(df.loc[0, :] - df.loc[1, :])
        df.loc['D13'] = abs(df.loc[0, :] - df.loc[2, :])
        df.loc['D21'] = abs(df.loc[1, :] - df.loc[0, :])
        df.loc['D23'] = abs(df.loc[1, :] - df.loc[2, :])
        df.loc['D31'] = abs(df.loc[2, :] - df.loc[0, :])
        df.loc['D32'] = abs(df.loc[2, :] - df.loc[1, :])
    discordance = pd.DataFrame([df.loc['D12'], df.loc['D13'], df.loc['D21'], df.loc['D23'], df.loc['D31'], df.loc['D32']])

    # find the max value across the whole row of the discordance data frame
    # Dmax1
    for row in discordance.iterrows():
        D12_abs_max = max(df.loc['D12'])
        D13_abs_max = max(df.loc['D13'])
        D21_abs_max = max(df.loc['D21'])
        D23_abs_max = max(df.loc['D23'])
        D31_abs_max = max(df.loc['D31'])
        D32_abs_max = max(df.loc['D32'])
    discordance_abs_max = pd.DataFrame([D12_abs_max, D13_abs_max, D21_abs_max, D23_abs_max, D31_abs_max, D32_abs_max])

    # create a discordance set where for each value that is 1 in concordance, the discordance value will replace it
    discordance_matrix_array = np.where(concordance == 0, discordance, 0)
    discordance_matrix = pd.DataFrame(discordance_matrix_array)

    # find the max value across the whole row of the discordance interval matrix
    # Dmax2
    D12_max = max(discordance_matrix.iloc[0, :])
    D13_max = max(discordance_matrix.iloc[1, :])
    D21_max = max(discordance_matrix.iloc[2, :])
    D23_max = max(discordance_matrix.iloc[3, :])
    D31_max = max(discordance_matrix.iloc[4, :])
    D32_max = max(discordance_matrix.iloc[5, :])
    discordance_max = pd.DataFrame([D12_max, D13_max, D21_max, D23_max, D31_max, D32_max])

    # Create the discordance interval matrix
    # divide Dmax2/Dmax1
    D12_maximum = D12_max/D12_abs_max
    D13_maximum = D13_max/D13_abs_max
    D21_maximum = D21_max/D21_abs_max
    D23_maximum = D23_max/D23_abs_max
    D31_maximum = D31_max/D31_abs_max
    D32_maximum = D32_max/D32_abs_max
    discordance_interval_matrix = (pd.DataFrame([[0, D12_maximum, D13_maximum], [D21_maximum,0, D23_maximum], [D31_maximum, D32_maximum,0]]))

    # Step 6: Find concordance index matrix
    # sum rows and columns for concordance interval matrix
    concordance_interval_matrix_row_sum = concordance_interval_matrix.sum(axis=1)

    # sum concordance interval matrix row sums
    concordance_interval_matrix_row_sums = concordance_interval_matrix_row_sum.sum()

    # c_bar = concordance_interval_matrix_row_sum / (number of alternatives(number of alternatives - 1))
    c_bar = concordance_interval_matrix_row_sums / (m * (m-1))

    # if concordance interval matrix value is less than c_bar, then concordance index matrix is 0, else 1
    concordance_index_matrix = np.where(concordance_interval_matrix.values < c_bar, 0, 1)

    # Step 7: Find discordance index matrix
    # find the sum of the rows for the discordance interval matrix
    discordance_interval_matrix_row_sum = discordance_interval_matrix.sum(axis=1)

    # sum concordance interval matrix row sums
    discordance_interval_matrix_row_sums = discordance_interval_matrix_row_sum.sum()

    # d_bar = discordance interval matrix / (number of alternatives (number of alternatives - 1))
    d_bar = discordance_interval_matrix_row_sums / (m*(m-1))

    # if discordance interval matrix value is greater than c_bar, then concordance index matrix is 0, else 1
    discordance_index_matrix = np.where(discordance_interval_matrix.values > d_bar, 0, 1)

    # Step 8: Calculate next superior and inferior values
    # for the concordance interval matrix, take the row sum - the column sum
    concordance_interval_matrix_col_sum = pd.DataFrame(concordance_interval_matrix.sum())
    concordance_interval_matrix_row_sum = pd.DataFrame(concordance_interval_matrix_row_sum)
    discordance_interval_matrix_col_sum = pd.DataFrame(discordance_interval_matrix.sum())
    discordance_interval_matrix_row_sum = pd.DataFrame(discordance_interval_matrix_row_sum)

    superior_values = pd.DataFrame(np.subtract(concordance_interval_matrix_row_sum, concordance_interval_matrix_col_sum)).transpose()
    inferior_values = pd.DataFrame(np.subtract(discordance_interval_matrix_row_sum, discordance_interval_matrix_col_sum)).transpose()

    superior_values_FINAL = pd.concat([superior_values_FINAL, superior_values]).reset_index(drop=True)
    inferior_values_FINAL = pd.concat([inferior_values_FINAL, inferior_values]).reset_index(drop=True)

    sheet_name_weight_scenario = pd.read_excel(data_path + '/criteria_weight_scenarios.xlsx',
                                                sheet_name='weight_scenarios').Scenario

    concordance_interval_matrix.to_excel(writer3, sheet_name=sheet_name_weight_scenario[i2])
    pd.DataFrame(concordance_index_matrix).to_excel(writer4, sheet_name=sheet_name_weight_scenario[i2])
    discordance_interval_matrix.to_excel(writer5, sheet_name=sheet_name_weight_scenario[i2])
    pd.DataFrame(discordance_index_matrix).to_excel(writer6, sheet_name=sheet_name_weight_scenario[i2])

superior_values_FINAL.columns = sanitation_systems
inferior_values_FINAL.columns = sanitation_systems

superior_values_FINAL = pd.concat([criteria_weight_scenario, criteria_weight_description, superior_values_FINAL], axis=1)
inferior_values_FINAL = pd.concat([criteria_weight_scenario, criteria_weight_description, inferior_values_FINAL], axis=1)

superior_values_FINAL.to_excel(writer2, sheet_name='superior_values')
inferior_values_FINAL.to_excel(writer2, sheet_name='inferior_values')

# writer2.save(), writer3.save(), writer4.save(), writer5.save(), writer6.save()


# AHP Methodology to Rank Alternatives #

# Output Excel File of Results
writer7 = pd.ExcelWriter(os.path.join(result_path, 'RESULTS_AHP.xlsx'))

AHP_Scores_FINAL = pd.DataFrame()
for i2 in range(num_weight):
    # Weighted Normalized Decision Matrix
    weighted_normal_matrix_FINAL = pd.DataFrame()
    for j in range(num_indicator):
        weighted_array = pd.DataFrame()
        indicator_criteria = indicator_type.iloc[0, j]
        if indicator_criteria == 'T':
            weight = criteria_weight.loc[i2, 'T'] * indicator_weight.iloc[0, j]
        elif indicator_criteria == 'RR':
            weight = criteria_weight.loc[i2, 'RR'] * indicator_weight.iloc[0, j]
        elif indicator_criteria == 'Env':
            weight = criteria_weight.loc[i2, 'Env'] * indicator_weight.iloc[0, j]
        elif indicator_criteria == 'Econ':
            weight = criteria_weight.loc[i2, 'Econ'] * indicator_weight.iloc[0, j]
        elif indicator_criteria == 'S':
            weight = criteria_weight.loc[i2, 'S'] * indicator_weight.iloc[0, j]
        else:
            print("Indicator Type Error")
        for k in range(num_system):
            weighted_normal_value = normal_matrix_FINAL.iloc[k, j] * weight
            weighted_normal_value = pd.DataFrame([weighted_normal_value])
            weighted_array = pd.concat([weighted_array, weighted_normal_value]).reset_index(drop=True)
        weighted_normal_matrix_FINAL = pd.concat([weighted_normal_matrix_FINAL, weighted_array], axis=1).reset_index(drop=True)
    weighted_normal_matrix_FINAL.columns = indicators

    AHP_Tech_Score = pd.DataFrame(weighted_normal_matrix_FINAL.iloc[:, 0:8].sum(axis=1)).transpose()
    AHP_RR_Score = pd.DataFrame(weighted_normal_matrix_FINAL.iloc[:, 9:14].sum(axis=1)).transpose()
    AHP_Env_Score = pd.DataFrame(weighted_normal_matrix_FINAL.iloc[:, 15:17].sum(axis=1)).transpose()
    AHP_Econ_Score = pd.DataFrame(weighted_normal_matrix_FINAL.iloc[:, 18]).transpose()
    AHP_Econ_Score = AHP_Econ_Score.reset_index(drop=True)
    AHP_Soc_Score = pd.DataFrame(weighted_normal_matrix_FINAL.iloc[:, 19:27].sum(axis=1)).transpose()

    AHP_Tech_Score.columns = ['T_A', 'T_B', 'T_C']
    AHP_RR_Score.columns = ['RR_A', 'RR_B', 'RR_C']
    AHP_Env_Score.columns = ['Env_A', 'Env_B', 'Env_C']
    AHP_Econ_Score.columns = ['Econ_A', 'Econ_B', 'Econ_C']
    AHP_Soc_Score.columns = ['S_A', 'S_B', 'S_C']

    AHP_Scores = pd.concat([AHP_Tech_Score, AHP_RR_Score, AHP_Env_Score, AHP_Econ_Score, AHP_Soc_Score], axis=1).reset_index(drop=True)

    AHP_Scores_FINAL = pd.concat([AHP_Scores_FINAL, AHP_Scores]).reset_index(drop=True)

AHP_Scores_FINAL = pd.concat([criteria_weight_scenario, criteria_weight_description, AHP_Scores_FINAL], axis=1)
AHP_Scores_FINAL.to_excel(writer7)

# writer7.save()