#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 07:46:08 2021

@author: Yalin Li <zoe.yalin.li@gmail.com>

Run this module to test whether an uncertainty parameter affects RR, Env, and Econ
score.
"""

import re
import numpy as np
import pandas as pd
from qsdsan.utils.decorators import time_printer
from dmsan.bwaise.sys_simulation import param_path, rebuild_models

models = modelA, modelB, modelC = rebuild_models()
dfs = dfA, dfB, dfC = [pd.read_excel(param_path, sheet_name=f'Alternative {i}', index_col=0)
                       for i in ('A', 'B', 'C')]


# %%

# Note that this will overwrite the "parameters.xlsx" file,
# so it's recommended to copy the RR, Env, and Econ columns to the
# "parameters_annotated.xlsx" file and do manual processing there
@time_printer
def test_param(save=True):
    def refresh_param(model): # some parameters will afect other parameters
        for p in model.parameters:
            p.setter(p.baseline)

    def format_dist(p):
        # regex = re.compile(r'(\w+)\(.*(\d*\.?\d*).*(\d*\.?\d*).*(\d*\.?\d*)\)')
        # mo = regex.search(str(p.distribution))
        # splitted = mo.groups()
        # splitted = re.split(r'\(|\)|,|=', str(p.distribution))
        # dist = list(splitted.pop(0))
        # dist.extend([i for i in splitted if i.isnumeric()])

        splitted = re.split(r'\(|\)|,|=', str(p.distribution))
        if splitted[0] in ('Uniform', 'Normal'):
            return splitted[0][0], float(splitted[2]), '', float(splitted[4])
        elif splitted[0] == 'Triangle':
            return 'T', float(splitted[2]), float(splitted[4]), float(splitted[6])
        else:
            raise ValueError('Distribution not uniform, triangular, or normal.')

    namesA = [p.name_with_units for p in modelA.parameters]
    namesB = [p.name_with_units for p in modelB.parameters]
    namesC = [p.name_with_units for p in modelC.parameters]

    for m, df in zip(models, dfs):
        m_baseline = m.metrics_at_baseline()
        for n, p in enumerate(m.parameters):
            alts = []
            for names in (namesA, namesB, namesC):
                alts.append(True if p.name_with_units in names else False)

            p_baseline = p.baseline

            # Min
            p.baseline = p.distribution.lower.item() if not p.hook \
                else p.hook(p.distribution.lower.item())
            refresh_param(m)
            m_min = m.metrics_at_baseline()

            # Max
            p.baseline = p.distribution.upper.item() if not p.hook \
                else p.hook(p.distribution.upper.item())
            refresh_param(m)
            m_max = m.metrics_at_baseline()

            p.baseline = p_baseline # reset parameter values
            diff = (m_max-m_min)/m_baseline
            diffs = [np.abs(diff[:3]).sum(), np.abs(diff[4:-1]).sum(), np.abs(diff[-1])]

            idx = df[df.Parameters==str((p.element_name, p.name_with_units))].index
            df.loc[idx, ['RR', 'Env', 'Econ']] = \
                [True if abs(i)>=1e-6 else False for i in diffs] # False for np.nan

            df.loc[idx, ['Baseline']] = p_baseline
            df.loc[idx, ['Distribution', 'Lower', 'Midpoint', 'Upper']] = \
                format_dist(p)
            df.loc[idx, ['Alternative A', 'Alternative B', 'Alternative C']] = alts

    if save:
        names = [f'Alternative {i}' for i in ('A', 'B', 'C')]
        with pd.ExcelWriter(param_path) as writer:
            for df, name in zip(dfs, names):
                df.to_excel(writer, sheet_name=name)

if __name__ == '__main__':
    test_param()