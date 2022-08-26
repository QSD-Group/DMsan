#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making for sanitation and resource recovery systems

This module is developed by:
    Yalin Li <mailto.yalin.li@gmail.com>
    Hannah Lohman <hlohman94@gmail.com>
'''

import os
from qsdsan.utils import load_pickle
from dmsan.utils import (
    init_modules,
    )

__all__ = (
    'scores_path',
    'results_path',
    'figures_path',
    'import_from_pickle',
    )

scores_path, results_path, figures_path = init_modules('bwaise')


def import_module_results(
        path, parameters=False, indicator_scores=False,
        ahp=False, mcda=False, uncertainty=False, sensitivity=None):
    loaded = dict.fromkeys(('params', 'tech_score', 'ahp', 'mcda',
                            'uncertainty', 'sensitivity'))

    if parameters:
        file_path = os.path.join(path, 'parameters.pckl')
        loaded['parameters'] = load_pickle(file_path)

    if indicator_scores:
        file_path = os.path.join(path, 'indicator_scores.pckl')
        loaded['indicator_scores'] = load_pickle(file_path)

    if ahp:
        file_path = os.path.join(path, 'ahp.pckl')
        loaded['ahp'] = load_pickle(file_path)

    if mcda:
        file_path = os.path.join(path, 'mcda.pckl')
        loaded['mcda'] = load_pickle(file_path)

    if uncertainty:
        file_path = os.path.join(path, 'uncertainty/performance_uncertainties.pckl')
        loaded['uncertainty'] = load_pickle(file_path)

    if sensitivity:
        file_path = os.path.join(path, f'sensitivity/performance_{sensitivity}_ranks.pckl')
        loaded['sensitivity'] = [load_pickle(file_path)]

        if sensitivity != 'KS':
            file_path = os.path.join(path, f'sensitivity/AHP_TOPSIS_{sensitivity}_scores.xlsx')
            loaded['sensitivity'].append(load_pickle(file_path))

    return loaded