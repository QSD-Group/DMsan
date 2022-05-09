#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Exposition of sanitation and resource recovery systems

This module is developed by:
    Hannah Lohman <hlohman94@gmail.com>
    Yalin Li <zoe.yalin.li@gmail.com>

'''

import os
from qsdsan.utils import load_pickle

biogenic_refinery_path = os.path.dirname(__file__)
scores_path = os.path.join(biogenic_refinery_path, 'scores')
results_path = os.path.join(biogenic_refinery_path, 'results')
figures_path = os.path.join(biogenic_refinery_path, 'figures')


def import_from_pickle(parameters=False, indicator_scores=False,
                       ahp=False, mcda=False,
                       uncertainty=False, sensitivity=None):
    loaded = dict.fromkeys(('param', 'tech_score', 'ahp', 'mcda',
                            'uncertainty', 'sensitivity'))

    if parameters:
        file_path = os.path.join(results_path, 'parameters.pckl')
        loaded['parameters'] = load_pickle(file_path)

    if indicator_scores:
        file_path = os.path.join(results_path, 'indicator_scores.pckl')
        loaded['indicator_scores'] = load_pickle(file_path)

    if ahp:
        file_path = os.path.join(results_path, 'ahp.pckl')
        loaded['ahp'] = load_pickle(file_path)

    if mcda:
        file_path = os.path.join(results_path, 'mcda.pckl')
        loaded['mcda'] = load_pickle(file_path)

    if uncertainty:
        file_path = os.path.join(results_path, 'uncertainty/performance_uncertainties.pckl')
        loaded['uncertainty'] = load_pickle(file_path)

    if sensitivity:
        file_path = os.path.join(results_path, f'sensitivity/performance_{sensitivity}_ranks.pckl')
        loaded['sensitivity'] = [load_pickle(file_path)]

        if sensitivity != 'KS':
            file_path = os.path.join(results_path, f'sensitivity/AHP_TOPSIS_{sensitivity}_scores.xlsx')
            loaded['sensitivity'].append(load_pickle(file_path))

    return loaded