# -*- coding: utf-8 -*-
"""
Modified on Wed Feb 24 08:48:08 2021

@author: torimorgan, hannahlohman, stetsonrowles
"""

                            ## MCDA Model ## 
#from __future__ import division #floating point division in Py2x
import numpy as np
#import math
#import pandas as pd


class MCDA:
                        ##DMsan values##   
    # Criteria
    #     Technical
    #     Environmental
    #     Economic
    #     Social
    # Indicators 
    #     Technical 
    #         resilience 
    #             loc_pop - current pop size of community [0-7]
    #             loc_training - the amount of training provided [0-7]
    #             loc_toilet_access - type of toilet * distance to next toilet [0-7]
    #         feasibility
    #             tech_absorption - to what extent do businesses adopt the latest
    #                   technologies [0-7]
    #             quality_road - local road infrastructure quality [0-7]
    #             tech_availability - to what extent are the latest technologies
    #                   available [0-7]
    #             loc_construct - the number of contstruction workers available 
    #             loc_OM - the number of professional skilled labors for O&M available
    #         flexibility
    #             loc_pop_growth - the impact on cost with population changes 
    #             loc_climate_risk - the impact on cost with climate changes
    #     Environmental
    #         #LCA endpoints
    #             LCA_ozone_depletion_endpoint
    #             LCA_global_warming_endpoint
    #             LCA_smog_endpoint
    #             LCA_acidification_endpoint
    #             LCA_eutrophication_LCA_endpoint
    #             LCA_carcinogenics_LCA_endpoint
    #             LCA_non-carcinogenics_LCA_endpoint
    #             LCA_respiratory_LCA_endpoint
    #             LCA_ecotoxicity_LCA_endpoint
    #             LCA_fossil_fuel_LCA_endpoint
    #         Resource Recovery Potential
    #             loc_drought_risk - the local drought risk 
    #             loc_fertilizer_need - the amount of fertilizer needed
    #             loc_renewable_energy - the amount of renewable energy 
    #                   infrastructure available
    #             loc_infrastructure
    #     Economic
    #         Net Annualized Cost
    #             loc_income - ??
    #     Social
    #         Job Creation
    #             loc_unemployment_rate - the local unemployment rate
    #         End-user Acceptability
    #             disposal_times_user - preference of community
    #             cleaning_hours_user - preference of community
    #             loc_privacy - preference of community
    #             odor - preference of community
    #             noise - noise produced by system
    #             visual - visual appearance of system
    #             loc_PPE - familiarity with PPE
    #             loc_safety - crime rate of the area
    #         Management Acceptability
    #             disposal_times_management - preference of manager
    #             cleaning_hours_management - preference of manager
    #             disposal_safety_management - PPE available for managers 
    # 37 indicators total 
                
    
## indicator parameters - import spreadsheet tabs as dataframes
# tech_availability = pd.read_excel('assumptions_DMsan.xlsx', sheet_name = 'AvailabilityLatestTech', index_col = 'Parameter')
#have a seperate file that would load all of the indicators for country
#index to country 
#give a code to each country to help with missspelling


## number of Monte Carlo runs
#n_samples = int(general_assumptions.loc['n_samples','expected'])
    
                    ##global and manual entries##

                    
##Bawaise Example 
    ##Alternatives   
        ##alt_1 is the baseline scenario where pit latrines shared by >3 
            #households, and tanker trucks convey slude from latrines to 
            #centralized treatment facilities. 
        ##alt_2 is the existing system but centralized treatment is replaced with
            #an anaerobic baffled reactor, solids drying beds, and an a
            #dditional planted bed for liquid treatment (designs and estimates
            #provided by CIDI
        ##alt_3 replaces pit latrines with container-based sanitation facilities,
            #storing source-separated urine and feces in easily removable 
            #containers that are collected frequently (2/week) by dedicated 
            #employees operating along defined collection routes. Employees 
            #use manually operated pushcarts to access facilities in densely 
            #populted settlements with limited road infrastructure. 

#   Numberof alterantives
    number_alternatives = 3

#   To pull in country specific data
    #location = Uganda    

#   This is a location parameter but requires manual entry
    #loc_pop = 

    
                      
       
#X is N/A as a default. The user must input values if data/expert judgement is available
#If data/expert judgement is not available, then leave value as X 

    #Criteria Indicators 
    #Technical 
        #resilience 
            #design_pop - current pop size input fluctuation that it can handle
                
    design_pop_alt_1 =  range(30000, 50000)
    design_pop_alt_2 =  range(30000, 50000)
    design_pop_alt_3 =  range(30000, 50000)
    
            #design_input - adaptability to inputs (paper, menstral products,etc.)
                    ##For definitions of these components, you can find them in the 
                    #compendium of sanitation systems and technologies 
                    #https://www.iwa-network.org/wp-content/uploads/2016/06/Compendium-Sanitation-Systems-and-Technologies.pdf
                
                # 1 - 7 with 1 being most simple and 7 being most complex
                #dry toilet = 1
                #pour flush toilet = 2
                #cisterian flush toilet = 3
                #UDDT = 4
                #UDFT = 5
                #Fossa Alterna = 6
                #Dehydration Vaults = 7
    
    design_input_alt_1 =  1  
    design_input_alt_2 =  1   
    design_input_alt_3 =  4  
       
            #treatment resilience is the ability for treatment to resist failure
                #physical = 1
                #thermal/chemical = 2
                #(biological) aerobic = 3
                #(biological) anaerobic = 4
                #examples in S.I. technology comparison to technology examples
                
                ###!!!!!!double check for if sedimentation is biological or physical 
                
    design_treatment = 1
    design_treatment = 4
    design_treatment = 1
   
        #feasibility
            #custom_parts - required custom parts for system
            #compendium 
            # 1 - 5 very low, edit this 
            
    custom_parts_alt_1 = 1
    custom_parts_alt_2 = 2
    custom_parts_alt_3 = 1
                
            #design_transport - the transporation required for system
                    #For definitions of these components, you can find them in the 
                    #compendium of sanitation systems and technologies 
                    #https://www.iwa-network.org/wp-content/uploads/2016/06/Compendium-Sanitation-Systems-and-Technologies.pdf
                # 1 - 7 with 1 being most simple and 7 being most complex
                #Jerrycan/Tank = 1
                #Human-Powered Emptying and Transport = 2
                #Motorized Emptying and Transport = 3
                #Simplified Sewer = 4
                #Solids-Free Sewer = 5
                #Conventional Gravity Sewer = 6 
                #Transfer Station = 7   
    design_transport_alt_1 = 3
    design_transport_alt_2 = 3
    design_transport_alt_3 = 2
    
    
            #design_construct - requirement for skilled labor 
            # 1 - 5 where 1 is existing and 3 is intensive labor and time for construction
                #For information on how difficult treatment construction can be, 
                #check the compendium or the excel reference document -> https://www.iwa-network.org/wp-content/uploads/2016/06/Compendium-Sanitation-Systems-and-Technologies.pdf

    design_construct_alt_1 = 1
    design_construct_alt_2 = 3
    design_construct_alt_3 = 1
    
            #design_OM - O&M complexity 
            # 1 - 3 where 1 is low, 2 is medium, 3 is a new technology 
   
    design_OM_alt_1 = 1
    design_OM_alt_2 = 2
    design_OM_alt_3 = 1
    
    #!!!!!!...Stetson could work on this section
        #flexibility 
            #design_pop_flex - the impact on cost with population changes
            # 1 - 3 where:
                # 1 is can handle less than 10% change in population, 
                # 2 is can handle 10 - 25% change in population, 
                # 3 is can handle more than 25% change in population.
                #how to handle decrease in population? 
                #metcalf and eddy safety factors
    # alt_1 designed for 50,000 and is serving 30,000        
    desing_pop_flex_alt_1 = 3
    # alt_2 and alt_3 designed for up to 55,000 and serving 45,000???
    desing_pop_flex_alt_2 = 2
    desing_pop_flex_alt_3 = 2 
    
        #flexibility to water table rise, wind damage, or flooding
        #for water table rise consider below ground components
        #for wind damage or flooding consider above ground componenets
        #  1 - 2 where:
            # 1 system has components underground or above ground
            # 2 system has components underground and above ground
            
        #  sum of costs to rebuild above and below ground components where:
            # [USD / cost to rebuild from water table rise]
            # [USD / cost to rebuild from flood/wind damage]  
            
        #  fraction of capital costs to rebuild above and below ground components after water table rise 1 m,
        #  by extreme/high wind speeds due to cyclonic activity (eg. wind damage from hurricanes, typhoons, and other wind activity),
        #  or flooding (overflowing of the normal confines of a stream or other body of water, or the accumulation of water over areas not normally submerged):
            # 1 is more than 50% captial costs required to rebuild 
            # 2 is 10 - 50% captial costs required to rebuild
            # 3 is 10% or less captial costs required to rebuild 
    # alt_1 has latrines (below ground) and fairly robust centralized treatment       
    design_climate_flex = 2
    # alt_2 has latrines (below ground) and fairly robust centralized treatment  
    design_climate_flex = 2
    # alt_3 has only above ground containment and fairly robust centralized treatment  
    design_climate_flex = 3
    
            #design_temp_flex - the impact of temperature on treatment
            #consider removing if double dipping with rel
            # 1 - 3 where:
                # 1 can handle no change in temperature (highly dependant on biological treatment), 
                # 2 can handle some change in temperature (dependant on biological treatment), 
                # 3 can handle significant changes in temperature (not dependant on biological treatment).
            
    desing_temp_flex_alt_1 = 2
    desing_temp_flex_alt_2 = 2
    desing_temp_flex_alt_3 = 2
            
            #design_energy_flex - energy demand and/or ability to withstand blackouts  
            #alt_2 higher energy usage
            
            #design_drought_flex - the impact of drought on treatment
            # 1 - 3 where:
                # 1 highly dependant on water, 
                # 2 some dependance on water, 
                # 3 no dependance on water.
                # user interface or constructed wetlands
       
    desing_drought_flex_alt_1 = 3
    desing_drought_flex_alt_2 = 3
    desing_drought_flex_alt_3 = 3

    
    #!!!!!!...Stetson could work on this section
                 
    #!!!!!....Yalin could work on this section
    #Environmental
        #LCA impact categories endpoint values
     #!!!!!....Yalin could work on this section
           

        #Resource Recovery Potential
            #water_recovery - percentage of water recovered
    water_recovery_alt_1 = 0
    water_recovery_alt_2 = 0
    water_recovery_alt_3 = 0
    
    #!!!!!Hannah will work on this section        
    #nutrient_recovery - normalized percentage of nutrients recovered
    #!!!!! nutrient_recovery_alt_1 = X
    #!!!!! nutrient_recovery_alt_2 = X
    #!!!!! nutrient_recovery_alt_3 = X
    #!!!!!Hannah will work on this section 
    
            #energy_recovery - energy recovered
    energy_recovery_alt_1 = 0
    energy_recovery_alt_2 = range(25, 55)
    energy_recovery_alt_3 = 0

            #design_infrastructure - the supply chain required to distribute resources 
            #number of resources recovered (the higher the better)
    design_infrastructure_alt_1 = 2
    design_infrastructure_alt_2 = 3
    design_infrastructure_alt_3 = 2
    
    #Economic
        #Net Annualized Cost
            #user_cost (USD·c/cap/·yr) - the net cost of the system per user per day
    user_cost_alt_1 = range(10,25)
    user_cost_alt_2 = range(3,18)
    user_cost_alt_3 = range(16, 36)

    #Social
        #Job Creation
            #total_job_creation - number of jobs 
            ##number of works for each alterantive
    #job_creation_alt_1 = 0
    #job_creation_alt_2 = 10
    #job_creation_alt_3 = 20,30 
            
            #higher_pay_jobs - number of higher payer jobs 
    #higher_pay_jobs_alt_1 = 0
    #higher_pay_jobs_alt_2 = 10
    #higher_pay_jobs_alt_3 = 0
    
        #End-user Acceptability
            #disposal_times_user - how many times disposed normalized to one year
    disposal_times_user_alt_1 = range(0.3, 2.4)
    disposal_times_user_alt_2 = range(0.3, 2.4)
    disposal_times_user_alt_3 = range(78, 130)
    
            #cleaning_hours_user - how many hours required of cleaning normalized to one year
            #1 - 5 where 1 is very low and 5 is very high (excel spreadsheet for more info)
    cleaning_hours_user_alt_1 = 2
    cleaning_hours_user_alt_2 = 2
    cleaning_hours_user_alt_3 = 4

            #number_households - how many household per system 
    number_households_alt_1 = 3
    number_households_alt_2 = 3
    number_households_alt_3 = 3
    
            #odor - odor produced by system
            #[5 = Pit latrine with dry toilet, 
            #4 = Ventilated Pit with dry toilet, 
            #3 = Pit Latrine with UDDT, 
            #2 = Ventilated Pit with UDDT, 
            #1 = any flush toilet and ventilated or container based storage] 
            #(1 = less odor, 5 = high odor)            
    odor_alt_1 = range(4,5)
    odor_alt_2 = range(4,5)
    odor_alt_3 = 1

#           noise - noise produced by system
    #noise_alt_1 = 1
    #noise_alt_2 = X
    #noise_alt_3 = X


#           design_PPE_user - PPE required for safe disposal 
    #design_PPE_user_alt_1 = X
    #design_PPE_user_alt_2 = X
    #design_PPE_user_alt_3 = X
    
#           system_distance - the distance from a household to a system 
    #system_distance_alt_1 = X
    #system_distance_alt_2 = X
    #system_distance_alt_3 = X
    
#       Management Acceptability
#           disposal_times_management - how many times disposed normalized to one year
    #disposal_times_management_alt_1 = X
    #disposal_times_management_alt_2 = X
    #disposal_times_management_alt_3 = X
    
#           cleaning_hours_management - how many hours required of cleaning 
    #cleaning_hours_management_alt_1 = X
    #cleaning_hours_management_alt_2 = X
    #cleaning_hours_management_alt_3 = X
    
#           design_PPE_management - PPE required for safe disposal 
    #design_PPE_management_alt_1 = X
    #design_PPE_management_alt_2 = X
    #design_PPE_management_alt_3 = X
            

    
#    number of indicators identified 
    #number_indicators = X

class AHP(MCDA):
    #def main(): 
#Criteria Matrix number_indicators x number_sindicators

    #       [ELI1      ELI2      ELI3     ELIn
    # ELI1    LI1/I1   LI1/LI2   LI1/I3  I1/In
    # ELI2    I2/I1    I2/I2   I2/I3  I2/In
    # ELI3    I3/I1   I3/I2   I3/I3  I3/In               
    # ELIn                           ]
    
#Criteria Weight Scale (Saaty et al., 1990)
#   Values    
#       w1 - 1.00 Equal Importance 
#       w2 - 2.00 Equal/Moderate Importance
#       w3 - 3.00 Moderate Importance
#       w4 - 4.00 Moderate/Strong Importance
#       w5 - 5.00 Strong Importancee 
#       w6 - 6.00 Strong/Very Strong Importance
#       w7 - 7.00 Very Strong Importance
#       w8 - 8.00 Very Strong/Extreme Importance
#       w9 - 9.00 Extreme Importance 
#   Intermediate Values
#       2,4,6,8 Intermediate Values
#   Inverse Values
#       w_half     - 1/2 = 0.500
#       w_third    - 1/3 = 0.333 
#       w_fourth   - 1/4 = 0.250
#       w_fifth    - 1/5 = 0.200
#       w_sixth    - 1/6 = 0.167
#       w_seventh  - 1/7 = 0.143 
#       w_eighth   - 1/8 = 0.125
#       w_ninth    - 1/9 = 0.111 
    
    w1 = 1.00
    w2 = 2.00
    w3 = 3.00
    w4 = 4.00
    w5 = 5.00
    w6 = 6.00
    w7 = 0.20
    w8 = 0.14 
    w9 = 0.11 
   
    w_half = 0.500
    w_third    - 1/3 = 0.333 
    w_fourth   - 1/4 = 0.250
    w_fifth    - 1/5 = 0.200
    w_sixth    - 1/6 = 0.167
    w_seventh  - 1/7 = 0.143 
    w_eighth   - 1/8 = 0.125
    w_ninth    - 1/9 = 0.111     
    
                 
    # Step 1: Assign criteria weights in matrix 
    
##Weighted Matrix Scenarios:## 

#W1 is when equal weights are given to all of the criteria    
# W1 = [[w1, w1, w1, w1], 
#       [w1, w1, w1, w1], 
#       [w1, w1, w1, w1], 
#       [w1, w1, w1, w1],] 
      
# W1_a = np.array(W1)

#Wn is when ....


#
 


                    
                ##Part A: Find Criteria Weights and Consistancy##   
# #Step 1: Sum the columns   
#     #sum of columns for Criteria Weight One Matrix
#     def column_sum(W1):
#         return[sum(i) for i in zip(*W1)]
#     global W1_col_sum 
#     W1_col_sum = np.array(column_sum(W1))
    
  
#     #Step 2: Normalize the matrix
#     W1_N = W1_a / W1_col_sum   
   
    
#     #Step 3: Calculate Criteria Weights by finding the row averages
#     CW_1=[sum(i) for i in W1_N]
    
    
#     #convert to an array
#     CW1_a = np.array(CW_1)
    
#     n=4    
#     #Find the average
#     CW1 = CW1_a / n
    
            
#                         #Step 4 Find the Consistancy ratio
#     #Step 4a: Calculate the weighted matrix by multiplying the matrix by the 
#         #criteria weight 
#     WM_1 = W1_a * CW1
    
#     #Step 4b: Sum the rows of the weighted matrix to find the weighted sum value
#     WS_1=[sum(i) for i in WM_1]
    
#     #convert to an array
#     WS1_a = np.array(WS_1)

    
#     #Step 4c: divide the weighted sum value by the criteria weights 
#     R_1 = WS1_a / CW1
    
#     #Step 4d: Find the Consistancy index by calculateing (delta max - n)/(n-1)
    
#     delta_max1 = (sum(R_1))/n
    
#     CI_1 = (delta_max1 - n) / (n - 1)

    
#     #Step 4e: Diovide the Consistancy index by the Random index 
    
#                                 #RI Values
#     #n=5;RI=1.12 n=6;RI=1.24 n=7;RI=1.32 n=8;RI=1.41 n=9;RI=1.45 n=10;RI=1.49
    
#     RI = 1.24
#     CR_1 = CI_1 / RI

    
    #If CR < 0.1 then our matrix is consistant 

        ##Part B: Find matrices of one criteria for each alternative##
    # for i in range(n_samples):
       
    #     def  __init__(self, S1_GHG, S1_PR, S1_NR, S1_CODR, S1_KR, S1_COST, 
    #                         S2_GHG, S2_PR, S2_NR, S2_CODR, S2_KR, S2_COST, S3_GHG, 
    #                        S3_PR, S3_NR, S3_CODR, S3_KR, S3_COST):   
    #         MCDA.__init__(self, S1_GHG, S1_PR, S1_NR, S1_CODR, S1_KR, S1_COST, 
    #                       S2_GHG, S2_PR, S2_NR, S2_CODR, S2_KR, S2_COST, S3_GHG, 
    #                       S3_PR, S3_NR, S3_CODR, S3_KR, S3_COST)
    #     #Step 1: Create pair wise comparison matrix    
    #     #       s1             s2                s3 
    #         self.GHG = [[1, (S2_GHG / S1_GHG), (S3_GHG / S1_GHG)],
    #                 [(S1_GHG / S2_GHG), 1, (S3_GHG / S2_GHG)],
    #                 [(S1_GHG / S3_GHG), (S2_GHG / S3_GHG), 1]]
    #         global GHG_a
    #         GHG_a = np.array(self.GHG)  
        
    #         self.PR =  [[1, (S1_PR / S2_PR), (S1_PR / S3_PR)],
    #                 [(S2_PR/ S1_PR), 1, (S2_PR / S3_PR)],
    #                 [(S3_PR / S1_PR), (S3_PR / S2_PR), 1]]
            
            
    #         global PR_a
    #         PR_a = np.array(self.PR)
            
    #         self.NR =  [[1, (S1_NR / S2_NR), (S1_NR / S3_NR)],
    #                 [(S2_NR/ S1_NR), 1, (S2_NR / S3_NR)],
    #                 [(S3_NR / S1_NR), (S3_NR / S2_NR), 1]]
            
    #         global NR_a
    #         NR_a = np.array(self.NR)
            
    #         self.CODR = [[1, (S1_CODR / S2_CODR), (S1_CODR / S3_CODR)],
    #                 [(S2_CODR/ S1_CODR), 1, (S2_CODR / S3_CODR)],
    #                 [(S3_CODR / S1_CODR), (S3_CODR / S2_CODR), 1]]
           
    #         global CODR_a
    #         CODR_a = np.array(self.CODR)
            
    #         self.KR = [[1, (S1_KR / S2_KR), (S1_KR / S3_KR)],
    #                 [(S2_KR/ S1_KR), 1, (S2_KR / S3_KR)],
    #                 [(S3_KR / S1_KR), (S3_KR / S2_KR), 1]]
            
    #         global KR_a
    #         KR_a = np.array(self.KR)
            
    #         self.COST = [[1, (S2_COST / S1_COST), (S3_COST / S1_COST)],
    #                 [(S1_COST / S2_COST), 1, (S3_COST / S2_COST)],
    #                 [(S1_COST / S3_COST), (S2_COST / S3_COST), 1]]
            
    #         global COST_a
    #         COST_a = np.array(self.COST)
            
    #         global mat_dict 
    #         mat_dict= {"GHG_a":GHG_a,"PR_a":PR_a,"NR_a":NR_a,"CODR_a":CODR_a,"KR_a":KR_a,"COST_a":COST_a}
    
    #         #Normalize Matrices 
            
    #         #sum of columns for GHG Matrix
        
    #     def column_sum(self, matrices_name):
    #         mat = mat_dict.get(matrices_name)
    #         return[sum(i) for i in zip(*mat)]
        
                
        
# ahp_object = AHP()   

# ahp_object.column_sum("GHG_a")    
# ahp_object.column_sum("PR_a")    
# ahp_object.column_sum("NR_a")    

class TOPSIS_MCDA(MCDA):
  
# #Step 1: Normalize matrix by finding the performance value 
#     def square(num):
#         return num ** 2
    
#     numbers = S_a
#     squares = list(map(square, numbers))
#     #print(squares)
    
#       #sum of columns^2 for alternative inputs
#     def column_sum(squares):
#         return[math.sqrt(sum(i)) for i in zip(*squares)]
#     global S_col_sum_sqrt
#     S_col_sum_sqrt = np.array(column_sum(squares))
#     #print(S_col_sum_sqrt)
    
#     #normalized perfromance value matrix
#     TN = (S_a / S_col_sum_sqrt)

# #Step 2: Multiply the weight by the normalized performance value
    
#     #weighted normalized decision matrix for scenario W1
#     TW_1 = [14.2857, 14.2857, 14.2857, 14.2857, 14.2857, 14.2857]
#     T_1 = TN * TW_1
#     T1 = np.array(T_1)
#     #print(T1)
    
   
#     #weighted normalized decision matrix for scenario W2
#     #eco is the economic criteria weight that is set 
#     eco = 70
#     w2 = (100 - eco)/5 
#     TW_2 = [w2, w2, w2, w2, w2, 70]
#     T_2 = TN * TW_2
#     T2 = np.array(T_2)
#     #print(T2)
    
    
#     #weighted when environmental is greater than economic
#     #env is the environmental criteria weight that is set
#     env = 70 
#     env_w = env / 5
#     w3 = (100 - env)
    
#     TW_3 = [env_w, env_w, env_w, env_w, env_w, w3]
#     T_3 = TN * TW_3
#     T3 = np.array(T_3)
#     print(T3)
    
       
# #Step 3: Indentify best and worst in column
   
#     #find min (ideal worst) and max (ideal best) of columns for T1
#     T1_col = np.matrix.transpose(T1)
#     T1_max =[max(i) for i in T1_col]
#     T1_min = [min(i) for i in T1_col]
#     #print(T1_max)
#     #print(T1_min)
    
#     #find min (ideal worst) and max (ideal best) of columns for T1
#     T2_col = np.matrix.transpose(T2)
#     T2_max =[max(i) for i in T2_col]
#     T2_min = [min(i) for i in T2_col]
#     #print(T2_max)
#     #print(T2_min)
    
#      #find min (ideal worst) and max (ideal best) of columns for T1
#     T3_col = np.matrix.transpose(T3)
#     T3_max =[max(i) for i in T3_col]
#     T3_min = [min(i) for i in T3_col]
#     print(T3_max)
#     print(T3_min)