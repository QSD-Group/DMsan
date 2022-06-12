#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making for sanitation and resource recovery systems

This module is developed by:
    Yalin Li <mailto.yalin.li@gmail.com>
'''
import os
path = os.path.dirname(__file__)
data_path = os.path.join(path, 'data')
from qsdsan.utils import load_pickle

from ._location import *
from ._ahp import *
from ._mcda import *

from . import (
    _location,
    _ahp,
    _mcda,
    )

def _init_modules(name):
    module_path = os.path.join(path, name)
    scores_path = os.path.join(module_path, 'scores')
    results_path = os.path.join(module_path, 'results')
    figures_path = os.path.join(module_path, 'figures')
    for p in (scores_path, results_path, figures_path):
        if not os.path.isdir(p): os.mkdir(p)
    return scores_path, results_path, figures_path

def _import_from_pickle(results_path,
                        parameters=False, indicator_scores=False,
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


__all__ = (
    *_location.__all__,
    *_ahp.__all__,
    *_mcda.__all__,
    'path',
    'data_path',
    '_init_modules',
    '_import_from_pickle',
    )