#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making for sanitation and resource recovery systems

This module is developed by:
    Yalin Li <mailto.yalin.li@gmail.com>
    Hannah Lohman <hlohman94@gmail.com>
'''

from dmsan import _init_modules, _import_from_pickle

scores_path, results_path, figures_path = _init_modules('bwaise')

def import_from_pickle(parameters=False, indicator_scores=False,
                       ahp=False, mcda=False,
                       uncertainty=False, sensitivity=None):
    return _import_from_pickle(
        results_path,
        parameters=parameters,
        indicator_scores=indicator_scores,
        ahp=ahp,
        mcda=mcda,
        uncertainty=uncertainty,
        sensitivity=sensitivity,
        )