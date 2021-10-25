==================================================
Case study of sanitation systems in Bwaise, Uganda
==================================================

.. contents::

Introduction
------------
TO BE ADDED


Structure
---------
TO BE ADDED


Instruction
-----------
To reproduce/regenerate the results, follow these steps:

#. Run `sys_simulation.py`, this will generate the baseline and uncertainty results for the technology scores simulated using QSDsan and EXPOsan (saved in ``scores``).
#. (Optionally) Run `param_testing.py`, this will run one-at-a-time local sensitivity analysis using the lower and upper bounds of the uncertainty parameters (saved in ``scores``), update ``parameters_annotated.xlsx`` using ``parameters.xlsx`` (both in ``scores``).
#. Run `uncertainty_sensitivity.py`, this will generate the uncertainty and sensitivity results by MCDA via AHP weighing and the TOPSIS method (saved in ``results``).
#. Run `ks_results_analysis.py`, this will analyze and illustrate results from the KS sensitivity analysis (saved in ``figures``).
#. Run `line_graph.py`, this will generate the line graphs (saved in ``figures``).
#. Run `scenario.py`, this will generate the graphs showing the increased winning chance with certian improvements (saved in ``figures``).