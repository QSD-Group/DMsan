#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 18:05:58 2021

@author: Yalin Li <zoe.yalin.li@gmail.com>
"""

# %%

import pandas as pd
import country_converter as coco
import os

# Data files associated with the model
data_path = os.path.join(os.path.dirname(__file__), 'data')

__all__ = ('Location', 'data_path')


# %%

class Location:
    '''Contains the contextual parameters related to a given location.'''

    def __init__(self, path='', location_name='Uganda'):
        path = path if path else data_path+'/location.xlsx'
        file = pd.ExcileFile(path)
        read_excel = lambda name: pd.read_excel(file, name, index_col='Country')
        self.location_name = coco.convert(names=[location_name], to='name_short')

        # Technical
        self.training = read_excel('ExtentStaffTraining') # T1
        self.sanitation_availability = read_excel('Sanitation') # T2
        self.tech_absorption = read_excel('TechAbsorption') # T3
        self.road_quality = read_excel('RoadQuality') # T4
        self.construction = read_excel('Construction') # T5
        self.OM_expertise = read_excel('AvailableScientistsEngineers') # T6
        self.pop_growth = read_excel('PopGrowth') # T7
        self.electricity_blackouts = read_excel('ElectricityBlackouts') # T8
        self.water_stress = read_excel('WaterStress') # T9

        # Resource Recovery Potential
        self.n_fertilizer_fulfillment = read_excel('NFertilizerFulfillment') # RR2
        self.p_fertilizer_fulfillment = read_excel('PFertilizerFulfillment') # RR3
        self.k_fertilizer_fulfillment = read_excel('KFertilizerFulfillment') # RR4
        self.renewable_energy = read_excel('RenewableEnergyConsumption') # RR5
        self.infrastructure = read_excel('InfrastructureQuality') # RR6

        # Social
        self.unemployment_rate = read_excel('UnemploymentTotal') # S1
        self.high_pay_jobs = read_excel('HighPayJobRate') # S2