#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yalin Li <zoe.yalin.li@gmail.com>
"""

import os
from qsdsan.utils import load_pickle

bwaise_path = os.path.dirname(__file__)
scores_path = os.path.join(bwaise_path, 'scores')
results_path = os.path.join(bwaise_path, 'results')
figures_path = os.path.join(bwaise_path, 'figures')

def import_from_pickle(param=False, tech_score=False, ahp=False, mcda=False,
                       uncertainty=False, sensitivity=None):
    loaded = dict.fromkeys(('param', 'tech_score', 'ahp', 'mcda',
                            'uncertainty', 'sensitivity'))

    if param:
        file_path = os.path.join(results_path, 'param.pckl')
        loaded['param'] = load_pickle(file_path)

    if tech_score:
        file_path = os.path.join(results_path, 'tech_score.pckl')
        loaded['tech_score'] = load_pickle(file_path)

    if ahp:
        file_path = os.path.join(results_path, 'ahp.pckl')
        loaded['ahp'] = load_pickle(file_path)

    if mcda:
        file_path = os.path.join(results_path, 'mcda.pckl')
        loaded['mcda'] = load_pickle(file_path)

    if uncertainty:
        file_path = os.path.join(results_path, 'uncertainty/AHP_TOPSIS.pckl')
        loaded['uncertainty'] = load_pickle(file_path)

    if sensitivity:
        file_path = os.path.join(results_path, f'sensitivity/AHP_TOPSIS_{sensitivity}_ranks.pckl')
        loaded['sensitivity'] = [load_pickle(file_path)]

        if sensitivity != 'KS':
            file_path = os.path.join(results_path, f'sensitivity/AHP_TOPSIS_{sensitivity}_scores.xlsx')
            loaded['sensitivity'].append(load_pickle(file_path))

    return loaded