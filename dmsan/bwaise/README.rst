==================================================
Case study of sanitation systems in Bwaise, Uganda
==================================================

Instructions
------------
To reproduce/regenerate the results, follow these steps:

#. Run `sys_simulation.py`, this will generate the baseline and uncertainty results for the technology scores simulated using QSDsan and EXPOsan (saved in ``scores``). It will also run one-at-a-time local sensitivity analysis using the lower and upper bounds of the uncertainty parameters (saved in ``scores``), then update ``parameters_annotated.xlsx`` using ``parameters.xlsx`` (both in ``scores``).
#. Run `mcda.py`, this will generate the uncertainty and sensitivity results by MCDA via AHP weighing and the TOPSIS method (saved in ``results``).
#. Run `ks_results_analysis.py`, this will analyze and illustrate results from the KS sensitivity analysis (saved in ``figures``).
#. Run `line_graph.py`, this will generate the line graphs (saved in ``figures``).
#. Run `improvements.py`, this will generate the graphs showing the increased winning chance with certain improvements (saved in ``results`` and ``figures``).