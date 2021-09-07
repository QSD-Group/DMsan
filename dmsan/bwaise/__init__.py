#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yalin Li <zoe.yalin.li@gmail.com>
"""

#!!! Consider using the pickle function in qsdsan when the new version is released
import os, pickle
bwaise_path = os.path.dirname(__file__)
scores_path = os.path.join(bwaise_path, 'scores')
results_path = os.path.join(bwaise_path, 'results')
figures_path = os.path.join(bwaise_path, 'figures')

def import_from_pickle(param=False, tech_score=False, ahp=False, mcda=False,
                       uncertainty=False, sensitivity=None):
    def load(path):
        f = open(path, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj

    loaded = dict.fromkeys(('param', 'tech_score', 'ahp', 'mcda', 
                            'uncertainty', 'sensitivity'))

    if param:
        file_path = os.path.join(results_path, 'param.pckl')
        loaded['param'] = load(file_path)

    if tech_score:
        file_path = os.path.join(results_path, 'tech_score.pckl')
        loaded['tech_score'] = load(file_path)

    if ahp:
        file_path = os.path.join(results_path, 'ahp.pckl')
        loaded['ahp'] = load(file_path)

    if mcda:
        file_path = os.path.join(results_path, 'mcda.pckl')
        loaded['mcda'] = load(file_path)

    if uncertainty:
        file_path = os.path.join(results_path, 'uncertainty/AHP_TOPSIS.pckl')
        loaded['uncertainty'] = load(file_path)

    if sensitivity:
        file_path = os.path.join(results_path, f'sensitivity/AHP_TOPSIS_{sensitivity}_ranks.pckl')
        loaded['sensitivity'] = [load(file_path)]

        if sensitivity != 'KS':
            file_path = os.path.join(results_path, f'sensitivity/AHP_TOPSIS_{sensitivity}_scores.xlsx')
            loaded['sensitivity'].append(load(file_path))

    return loaded