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

import os
import pandas as pd
import country_converter as coco
from . import data_path

__all__ = ('Location',)


# %%

class Location:
    '''Contains the contextual parameters related to a given location.'''

    def __init__(self, file_path='', location_name='Uganda'):
        path = file_path if file_path else os.path.join(data_path, 'location.xlsx')
        file = pd.ExcelFile(path)
        read_excel = lambda name: pd.read_excel(file, name, index_col='Country') # name is sheet name
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

    def __repr__(self):
        return f'<Location: {self.location_name}>'