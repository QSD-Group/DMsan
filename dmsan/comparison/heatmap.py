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
from chaospy import distributions as shape
from dmsan.comparison import results_path

N = 1000
seed = 3221
br_wage_bounds = (1.0015, 336.9715)
re_ng_wage_bounds = (0.1251875, 42.1214375)
electricity_price_bounds = (0.003, 0.378)
e_cal_bounds = (1786.0, 3885.0)
p_anim_bounds = (6.55, 104.98)
p_veg_bounds = (24.81, 73.29)
electricity_gwp_bounds = (0.012, 1.046968)


# %%

# Biogenic Refinery, sysA
from exposan import biogenic_refinery as br
br.INCLUDE_RESOURCE_RECOVERY = False
from exposan.biogenic_refinery import create_model as create_br_model
brA_model = create_br_model('A')

for wage_param in brA_model.parameters:
    if wage_param.name in ('Operator daily wages', 'Operator daily wage'): break
wage_param.distribution = shape.Uniform(*br_wage_bounds) # lower, upper

for electricity_price_param in brA_model.parameters:
    if electricity_price_param.name in ('Energy price', 'Electricity price'): break
electricity_price_param.distribution = shape.Uniform(*electricity_price_bounds)

brA_model.parameters = [wage_param, electricity_price_param]
brA_model.load_samples(brA_model.sample(N=N, rule='L', seed=seed))

brA_model.evaluate()

brA_model.table.to_excel(os.path.join(results_path, 'heatmap_brA.xlsx'))


# NEWgen, sysB
from exposan import new_generator as ng
ng.INCLUDE_RESOURCE_RECOVERY = False
from exposan.new_generator import create_model as create_ng_model
ngB_model = create_ng_model('B')

for wage_param in ngB_model.parameters:
    if wage_param.name in ('Wages', 'Labor wages'): break
wage_param.distribution = shape.Uniform(*re_ng_wage_bounds)

for electricity_price_param in ngB_model.parameters:
    if electricity_price_param.name in ('Energy price', 'Electricity price'): break
electricity_price_param.distribution = shape.Uniform(*electricity_price_bounds)

ngB_model.parameters = [wage_param, electricity_price_param]
ngB_model.load_samples(ngB_model.sample(N=N, rule='L', seed=seed))
ngB_model.evaluate()

ngB_model.table.to_excel(os.path.join(results_path, 'heatmap_ngB.xlsx'))


# Reclaimer, sysB
from exposan import reclaimer as re
re.INCLUDE_RESOURCE_RECOVERY = False
from exposan.reclaimer import create_model as create_re_model
reB_model = create_re_model('B')

for wage_param in reB_model.parameters:
    if wage_param.name in ('Wages', 'Labor wages'): break
wage_param.distribution = shape.Uniform(*re_ng_wage_bounds)

for electricity_price_param in reB_model.parameters:
    if electricity_price_param.name in ('Energy price', 'Electricity price'): break
electricity_price_param.distribution = shape.Uniform(*electricity_price_bounds)

reB_model.parameters = [wage_param, electricity_price_param]
reB_model.load_samples(reB_model.sample(N=N, rule='L', seed=seed))
reB_model.evaluate()

reB_model.table.to_excel(os.path.join(results_path, 'heatmap_reB.xlsx'))


# Biogenic Refinery, sysB
from exposan import biogenic_refinery as br
br.INCLUDE_RESOURCE_RECOVERY = False
from exposan.biogenic_refinery import create_model as create_br_model
brB_model = create_br_model('B')

for e_cal_param in brB_model.parameters:
    if e_cal_param.name in ('Excretion e cal', 'E cal'): break
e_cal_param.distribution = shape.Uniform(*e_cal_bounds)

for p_anim_param in brB_model.parameters:
    if p_anim_param.name in ('Excretion p anim', 'P anim'): break
p_anim_param.distribution = shape.Uniform(*p_anim_bounds)

brB_model.parameters = [e_cal_param, p_anim_param]
brB_model.load_samples(brB_model.sample(N=N, rule='L', seed=seed))
brB_model.evaluate()

brB_model.table.to_excel(os.path.join(results_path, 'heatmap_brB.xlsx'))


# NEWgen, sysA
from exposan import new_generator as ng
ng.INCLUDE_RESOURCE_RECOVERY = False
from exposan.new_generator import create_model as create_ng_model
ngA_model = create_ng_model('A')

for e_cal_param in ngA_model.parameters:
    if e_cal_param.name in ('Excretion e cal', 'E cal'): break
e_cal_param.distribution = shape.Uniform(*e_cal_bounds)

for p_anim_param in ngA_model.parameters:
    if p_anim_param.name in ('Excretion p anim', 'P anim'): break
p_anim_param.distribution = shape.Uniform(*p_anim_bounds)

ngA_model.parameters = [e_cal_param, p_anim_param]
ngA_model.load_samples(ngA_model.sample(N=N, rule='L', seed=seed))
ngA_model.evaluate()

ngA_model.table.to_excel(os.path.join(results_path, 'heatmap_ngA.xlsx'))


# Reclaimer, sysC
from exposan import reclaimer as re
re.INCLUDE_RESOURCE_RECOVERY = False
from exposan.reclaimer import create_model as create_re_model
reC_model = create_re_model('C')

for e_cal_param in reC_model.parameters:
    if e_cal_param.name in ('Excretion e cal', 'E cal'): break
e_cal_param.distribution = shape.Uniform(*e_cal_bounds)

for p_anim_param in reC_model.parameters:
    if p_anim_param.name in ('Excretion p anim', 'P anim'): break
p_anim_param.distribution = shape.Uniform(*p_anim_bounds)

reC_model.parameters = [e_cal_param, p_anim_param]
reC_model.load_samples(reC_model.sample(N=N, rule='L', seed=seed))
reC_model.evaluate()

reC_model.table.to_excel(os.path.join(results_path, 'heatmap_reC.xlsx'))


# Alternative GHG graphs

# Biogenic Refinery, sysB
from exposan import biogenic_refinery as br
br.INCLUDE_RESOURCE_RECOVERY = False
from exposan.biogenic_refinery import create_model as create_br_model
brB_model = create_br_model('B')

for p_veg_param in brB_model.parameters:
    if p_veg_param.name in ('Excretion p veg', 'P veg'): break
p_veg_param.distribution = shape.Uniform(*p_veg_bounds)

for p_anim_param in brB_model.parameters:
    if p_anim_param.name in ('Excretion p anim', 'P anim'): break
p_anim_param.distribution = shape.Uniform(*p_anim_bounds)

brB_model.parameters = [p_veg_param, p_anim_param]
brB_model.load_samples(brB_model.sample(N=N, rule='L', seed=seed))
brB_model.evaluate()

brB_model.table.to_excel(os.path.join(results_path, 'heatmap_brB_pveg_panim.xlsx'))


# NEWgen, sysB
from exposan import new_generator as ng
ng.INCLUDE_RESOURCE_RECOVERY = False
from exposan.new_generator import create_model as create_ng_model
ngB_model = create_ng_model('B')

for e_cal_param in ngB_model.parameters:
    if e_cal_param.name in ('Excretion e cal', 'E cal'): break
e_cal_param.distribution = shape.Uniform(*e_cal_bounds)

for electricity_gwp_param in ngB_model.parameters:
    if electricity_gwp_param.name in ('Electricity CF', 'Energy gwp'): break
electricity_gwp_param.distribution = shape.Uniform(*electricity_gwp_bounds)

ngB_model.parameters = [e_cal_param, electricity_gwp_param]
ngB_model.load_samples(ngB_model.sample(N=N, rule='L', seed=seed))
ngB_model.evaluate()

ngB_model.table.to_excel(os.path.join(results_path, 'heatmap_ngB_ecal_electricity_gwp.xlsx'))


# Reclaimer, sysB
from exposan import reclaimer as re
re.INCLUDE_RESOURCE_RECOVERY = False
from exposan.reclaimer import create_model as create_re_model
reB_model = create_re_model('B')

for e_cal_param in reB_model.parameters:
    if e_cal_param.name in ('Excretion e cal', 'E cal'): break
e_cal_param.distribution = shape.Uniform(*e_cal_bounds)

for electricity_gwp_param in reB_model.parameters:
    if electricity_gwp_param.name in ('Electricity CF', 'Energy gwp'): break
electricity_gwp_param.distribution = shape.Uniform(*electricity_gwp_bounds)

reB_model.parameters = [e_cal_param, electricity_gwp_param]
reB_model.load_samples(reB_model.sample(N=N, rule='L', seed=seed))
reB_model.evaluate()

reB_model.table.to_excel(os.path.join(results_path, 'heatmap_reB_ecal_electricitygwp.xlsx'))
