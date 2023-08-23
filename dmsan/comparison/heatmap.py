# -*- coding: utf-8 -*-

'''
DMsan: Decision-making of sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>
    
    Hannah Lohman <hlohman94@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/DMsan/blob/main/LICENSE.txt
for license details.

Run this module to generate the data needed for making the heatmap.
'''

import os
from dmsan.comparison import results_path

N = 10
seed = 3221
br_wage_bounds = (1.0015, 336.9715)
re_ng_wage_bounds = (0.1251875, 42.1214375)
electricity_price_bounds = (0.003, 0.378)


# %%

# Biogenic Refinery, sysA
from exposan import biogenic_refinery as br
br.INCLUDE_RESOURCE_RECOVERY = False
from exposan.biogenic_refinery import create_model as create_br_model
brA_model = create_br_model('A')

for wage_param in brA_model.parameters:
    if wage_param.name in ('Operator daily wages', 'Operator daily wage'): break
wage_param.bounds = br_wage_bounds

for electricity_price_param in brA_model.parameters:
    if electricity_price_param.name in ('Energy price', 'Electricity price'): break
electricity_price_param.bounds = electricity_price_bounds

brA_model.parameters = [wage_param, electricity_price_param]
brA_model.load_samples(brA_model.sample(N=N, rule='L', seed=seed))
brA_model.evaluate()

brA_model.table.to_excel(os.path.join(results_path, 'heatmap_brA.xlsx'))
