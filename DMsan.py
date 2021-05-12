# -*- coding: utf-8 -*-
"""
Modified on Tue May 4

@author: torimorgan, hannahlohman, stetsonrowles, yalinli
"""

                            ## MCDA Model ## 
#from __future__ import division #floating point division in Py2x
import numpy as np
#import math
import pandas as pd
import country_converter as coco



class MCDA:
                                  
#Step 1: Identify Location

##Input location by country in the brackets ##
    location = ['Uganda']    

#converting location to match the database
    location = coco.convert(names=location, to='name_short')

#mannual inputs defined here 

X = 0.00001

#Criteria: Social
#Subcriteria: End-user acceptability
##Input community preference ## 
#Local Weight Indicator S3: Disposal convenience preference for user
#realtes to the preference for disposal requirements on the user end
#if management is responsible for disposal, then insert X
# 0 being low preference to frequency of disposal to 100 being 
    #high preference for frequency of disposal
s3 = X

## Input community preference ##
#Local Weight Indicator S4: Cleaning preference
#realtes to the preference for cleaning requirements
# 0 being low preference to frequency of cleaning to 100 being 
    #high preference for frequency of cleaning
s4 = 44

## Input community preference ##
#Local Weight Indicator S5: Privacy preference
#realtes to the preference for privacy (# of households sharing a system)
# 0 being low preference for privacy to 100 being 
    #high preference for privacy
s5 = 47

## Input community preference ##
#Local Weight Indicator S6: Odor preference
#realtes to the preference of odor with 
# 0 being low preference for odor to 100 being high preference for odor
s6 = 22

## Input community preference ##
#Local Weight Indicator S7: Noise preference
#realtes to the preference of noise with 
# 0 being low preference for odor to 100 being high preference for noise
s7 = X

## Input community preference ##
#Local Weight Indicator S8: PPE preference/familiarity
#realtes to the preference of PPE with 
# 0 being low importance for PPE to 100 being high importance for PPE
s8 = X

## Input community preference ##
#Local Weight Indicator S9: Security preference
#realtes to the preference of security with 
# 0 being low preference for secutiy to 100 being high preference for odor
s9 = X

#Subcriteria: Management Acceptability
## Input management (i.e., landlord) preference ##
#Local Weight Indicator S10: Disposal convenience preference
#realtes to the preference for disposal requirements 
# 0 being low importance to frequency of disposal to 100 being 
    #high importance for frequency of disposal
s10 = X

## Input management preference ##
#Local Weight Indicator S11: Cleaning preference
#realtes to the preference for cleaning requirements 
# 0 being low importance to frequency of cleaning to 100 being 
    #high importance for frequency of cleaning
s11 = X

## Input management preference ##
#Local Weight Indicator S12: PPE preference/familiarity
#realtes to the preference of PPE with 
# 0 being low importance for PPE to 100 being high importance for PPE
s12 = X


                            #Local Weights#
#Criteria: Technical
#Subcriteria: Resilience                           
#Local Weight Indicator T1: Extent of training 
#realtes to how much training is available to train users and personnel
training = pd.read_excel('/Users/torimorgan/Desktop/DMsan/location.xlsx', 
                                   sheet_name = 'ExtentStaffTraining', index_col= 'Country') 
t1 = (training.loc[location, 'Value'])
T1 = (100 - (t1/7*100))

#Local Weight Indicator T2: Population with access to imporved sanitation
#realtes to how available improved sanitation is in the region in case a system fails
sanitation_availability = pd.read_excel('/Users/torimorgan/Desktop/DMsan/location.xlsx', 
                                   sheet_name = 'Sanitation', index_col= 'Country') 
t2 = (sanitation_availability.loc[location, 'Value'])
T2 = (100 - t2)

#Subcriteria: Feasibility
#Local Weight Indicator T3: Accessibility to technology 
#realtes to how easily the region can access technology 
tech_absorption = pd.read_excel('/Users/torimorgan/Desktop/DMsan/location.xlsx', 
                                   sheet_name = 'TechAbsorption', index_col= 'Country') 
t3 = (tech_absorption.loc[location, 'Value'])
T3 = (100-(t3/7*100))

#Local Weight Indicator T4: Transporation infrastructure
#realtes to the quality of transporation infrastructure for transport of waste
road_quality = pd.read_excel('/Users/torimorgan/Desktop/DMsan/location.xlsx', 
                                   sheet_name = 'RoadQuality', index_col= 'Country') 
t4 = (road_quality.loc[location, 'Value'])
T4 = (100-(t4/7*100))

#Local Weight Indicator T5: Construction skills available
#realtes to the construction expertise available
construction = pd.read_excel('/Users/torimorgan/Desktop/DMsan/location.xlsx', 
                                   sheet_name = 'Construction', index_col= 'Country') 
t5 = (construction.loc[location, 'Value'])
T5 = (100 - (t5/40.5*100))

#Local Weight Indicator T6: O&M experitse available
#realtes to the O&M expertise available
OM_expertise = pd.read_excel('/Users/torimorgan/Desktop/DMsan/location.xlsx', 
                                   sheet_name = 'AvailableScientistsEngineers', index_col= 'Country') 
t6 = (OM_expertise.loc[location, 'Value'])
T6 = (100-(t6/7*100))
   
#Local Weight Indicator T7: Population growth trajectory
#realtes to the population flexibility 
pop_growth = pd.read_excel('/Users/torimorgan/Desktop/DMsan/location.xlsx', 
                                   sheet_name = 'PopGrowth', index_col= 'Country') 
t7 = (pop_growth.loc[location, 'Value'])
T7 = (t7/4.5*100)

#Local Weight Indicator T8: 
#realtes to the climate risk 1
climate_risk_1 = pd.read_excel('/Users/torimorgan/Desktop/DMsan/location.xlsx', 
                                   sheet_name = 'ClimateRisk', index_col= 'Country') 
t8 = (climate_risk_1.loc[location, 'Value'])
T8 = (100-(t6/7*100))

#Local Weight Indicator T9:
#realtes to the climate risk 2
climate_risk_1 = pd.read_excel('/Users/torimorgan/Desktop/DMsan/location.xlsx', 
                                   sheet_name = 'ClimateRisk', index_col= 'Country') 
t9 = 60
T9 = 60

#Local Weight Indicator T10:
#realtes to the climate risk 3
climate_risk_1 = pd.read_excel('/Users/torimorgan/Desktop/DMsan/location.xlsx', 
                                   sheet_name = 'ClimateRisk', index_col= 'Country') 
t10 = 80
T10 = 80

T11 = 60
T12 = 50

###Fix with Stetson's code



#Criteria: Environmental
#Subcriteria: LCA
#!!!!! #Yalin 

#Subcriteria: Resource Recovery Potential 
#!!!!! #Hannah

#Criteria: Social
#Subcriteria: Job Creation 
#Local Weight Indicator S1: Unemployment 
#realtes to the unemployment rate
unemployment_rate = pd.read_excel('/Users/torimorgan/Desktop/DMsan/location.xlsx', 
                                   sheet_name = 'UnemploymentTotal', index_col= 'Country') 
s1 = (unemployment_rate.loc[location, 'Value'])
S1 = (s1/28.74*100)

#Local Weight Indicator S2: High paying jobs
#realtes to the need for higher paying jobs
high_pay_jobs = pd.read_excel('/Users/torimorgan/Desktop/DMsan/location.xlsx', 
                                   sheet_name = 'HighPayJobRate', index_col= 'Country') 
s2 = (high_pay_jobs.loc[location, 'Value'])
S2 = (s2/94.3*100)

#Subcriteria
#Local Weight Indicator S3: Disposal convenience preference
#realtes to the preference for disposal requirements 
# 0 being low importance to frequency of disposal to 100 being 
    #high importance for frequency of disposal
S3 = s3

## Input community preference ##
#Local Weight Indicator S4: Cleaning preference
#realtes to the preference for cleaning requirements
# 0 being low importance to frequency of cleaning to 100 being 
    #high importance for frequency of cleaning
S4 = s4

## Input community preference ##
#Local Weight Indicator S5: Privacy preference
#realtes to the preference for privacy (# of households sharing a system)
# 0 being low importance for privacy to 100 being 
    #high importance for privacy
S5 = s5

## Input community preference ##
#Local Weight Indicator S6: Odor preference
#realtes to the preference of odor with 
# 0 being low importance for odor to 100 being high importance for odor
S6 = s6

## Input community preference ##
#Local Weight Indicator S7: Noise preference
#realtes to the preference of noise with 
# 0 being low importance for odor to 100 being high importance for noise
S7 = s7

## Input community preference ##
#Local Weight Indicator S8: PPE preference/familiarity
#realtes to the preference of PPE with 
# 0 being low importance for PPE to 100 being high importance for PPE
S8 = s8

## Input community preference ##
#Local Weight Indicator S9: Security preference
#realtes to the preference of security with 
# 0 being low importance for secutiy to 100 being high importance for odor
S9 = s9

#Subcriteria: Management Acceptability
## Input management (i.e., landlord) preference ##
#Local Weight Indicator S10: Disposal convenience preference
#realtes to the preference for disposal requirements 
# 0 being low importance to frequency of disposal to 100 being 
    #high importance for frequency of disposal
S10 = s10

## Input management preference ##
#Local Weight Indicator S11: Cleaning preference
#realtes to the preference for cleaning requirements 
# 0 being low importance to frequency of cleaning to 100 being 
    #high importance for frequency of cleaning
S11 = s11

## Input management preference ##
#Local Weight Indicator S12: PPE preference/familiarity
#realtes to the preference of PPE with 
# 0 being low importance for PPE to 100 being high importance for PPE
S12 = s12
                    
                    #Tech/System Performance Scores# 
class AHP(MCDA):
    def main(): 
                  
# Step 1: Assign criteria weights in matrix 
    
        T_W = [[T1/T1, T1/T2, T1/T3, T1/T4, T1/T5, T1/T6, T1/T7, T1/T8, T1/T9, T1/T10, T1/T11, T1/T12], 
         [T2/T1, T2/T2, T2/T3, T2/T4, T2/T5, T2/T6, T2/T7, T2/T8, T2/T9, T2/T10, T2/T11, T2/T12], 
         [T3/T1, T3/T2, T3/T3, T3/T4, T3/T5, T3/T6, T3/T7, T3/T8, T3/T9, T3/T10, T3/T11, T3/T12], 
         [T4/T1, T4/T2, T4/T3, T4/T4, T4/T5, T4/T6, T4/T7, T4/T8, T4/T9, T4/T10, T4/T11, T4/T12],
         [T5/T1, T5/T2, T5/T3, T5/T4, T5/T5, T5/T6, T5/T7, T5/T8, T5/T9, T5/T10, T5/T11, T5/T12],
         [T6/T1, T6/T2, T6/T3, T6/T4, T6/T5, T6/T6, T6/T7, T6/T8, T6/T9, T6/T10, T6/T11, T6/T12],
         [T7/T1, T7/T2, T7/T3, T7/T4, T7/T5, T7/T6, T7/T7, T7/T8, T7/T9, T7/T10, T7/T11, T7/T12],
         [T8/T1, T8/T2, T8/T3, T8/T4, T8/T5, T8/T6, T8/T7, T8/T8, T8/T9, T8/T10, T8/T11, T8/T12],
         [T9/T1, T9/T2, T9/T3, T9/T4, T9/T5, T9/T6, T9/T7, T9/T8, T9/T9, T9/T10, T9/T11, T9/T12],
         [T10/T1, T10/T2, T10/T3, T10/T4, T10/T5, T10/T6, T10/T7, T10/T8, T10/T9, T10/T10, T10/T11, T10/T12]] 
          
        T_W_a = np.array(T_W)
                    
                ##Part A: Find Criteria Weights and Consistancy##   
#Step 1: Sum the columns   
 #sum of columns for Criteria Weight One Matrix
    def column_sum(T_W):
        return[sum(i) for i in zip(*T_W)]
    global T_W_col_sum 
    T_W_col_sum = np.array(column_sum(T_W))    
  
#Step 2: Normalize the matrix
T_W_N = T_W_a / T_W_col_sum
   

#Step 3: Calculate Criteria Weights by finding the row averages
C_T_W =[sum(i) for i in T_W_N]

#convert to an array
C_T_W_a = np.array(C_T_W)

#number of indicators    
n=10 
 
#Find the average
Avg_C_T_W = C_T_W_a / n
            
#Step 4 Find the Consistancy ratio
#Step 4a: Calculate the weighted matrix by multiplying the matrix by the 
        #criteria weight 
WM_T = T_W_a * C_T_W 

#Step 4b: Sum the rows of the weighted matrix to find the weighted sum value
WS_T =[sum(i) for i in WM]

#convert to an array
WS_T_a = np.array(WS_T)

    
#Step 4c: divide the weighted sum value by the criteria weights 
R_T = WS1_a / CW1

#Step 4d: Find the Consistancy index by calculateing (delta max - n)/(n-1)   
delta_maxT = (sum(R_T))/n

CI_T = (delta_maxT - n) / (n - 1)


#Step 4e: Diovide the Consistancy index by the Random index 

                            #RI Values
#n=5;RI=1.12 n=6;RI=1.24 n=7;RI=1.32 n=8;RI=1.41 n=9;RI=1.45 n=10;RI=1.49
#If CR < 0.1 then our matrix is consistant 
RI = 1.24
CR_T = CI_T / RI


  

