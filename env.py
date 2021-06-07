#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 13:31:44 2021

@author: Yalin Li <zoe.yalin.li@gmail.com>

TODO: figure out why ReCiPe gives so much credit to fertilizers

"""

from exposan import bwaise as bw
bw.update_lca_data('new')

lcas = lcaA, lcaB, lcaC = bw.lcaA, bw.lcaB, bw.lcaC
get_ppl = bw.systems.get_ppl

indicators = []
for cat in ('I', 'H', 'E'):
    temp = [i for i in lcaA.indicators if cat+'_' in i.ID and '_Total' in i.ID]
    temp.sort(key=lambda ind: ind.ID)
    indicators.extend(temp)

for lca in lcas:
    lca.indicators = indicators

def get_annualized_pts(lca):
    impact_dct = lca.get_total_impacts()
    ratio = lca.lifetime * get_ppl(lca.system.ID[-1])
    for k, v in impact_dct.items():
        impact_dct[k] = v / ratio
    return impact_dct