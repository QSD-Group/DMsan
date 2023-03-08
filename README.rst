====================================================================================
DMsan: Decision-Making for Sanitation and Resource Recovery Systems
====================================================================================

.. License
.. image:: https://img.shields.io/pypi/l/qsdsan?color=blue&logo=UIUC&style=flat
   :target: https://github.com/QSD-Group/QSDsan/blob/main/LICENSE.txt

.. Python version
.. image:: https://img.shields.io/pypi/pyversions/dmsan?style=flat
   :target: https://pypi.python.org/pypi/dmsan

.. PyPI version
.. image:: https://img.shields.io/pypi/v/dmsan?style=flat&color=blue
   :target: https://pypi.org/project/dmsan

|

.. contents::

|

What is ``DMsan``?
-------------------
``DMsan`` is an open-source platform for decision-making for sanitation and resource recovery systems. It is one of a series of platforms that are being developed for the execution of QSD - a methodology for the research, design, and deployment of technologies and inform decision-making [1]_.

As an open-source multi-criteria decision analysis platform, DMsan enables users to transparently compare sanitation and resource recovery alternatives and characterize the opportunity space for early-stage technologies using multiple dimensions of sustainability with consideration of location-specific contextual parameters.

DMsan integrates with the open-source Python package `QSDsan <https://github.com/QSD-Group/QSDsan>`_ (the quantitative sustainable design (QSD) of sanitation and resource recovery systems) for system design and simulation to calculate quantitative economic (via techno-economic analysis, TEA), environmental (via life cycle assessment, LCA), and resource recovery indicators under uncertainty [2]_.

All systems developed with QSDsan are included in the package `EXPOsan <https://github.com/QSD-Group/EXPOsan>`_ - exposition of sanitation and resource recovery systems, which can be used to develop sanitation and resource recovery alternatives for evaluation in DMsan.


Installation
------------
The easiest way is through ``pip``, in command-line interface (e.g., Anaconda prompt, terminal):

.. code::

    pip install dmsan

If you need to upgrade:

.. code::

    pip install -U dmsan

or for a specific version (replace X.X.X with the version number):

.. code::

    pip install dmsan==X.X.X

If you want to install the latest GitHub version at the `main branch <https://github.com/qsd-group/dmsan>`_ (note that you can still use the ``-U`` flag for upgrading):

.. code::

    pip install git+https://github.com/QSD-Group/DMsan.git


.. note::

   If this doesn't give you the newest ``dmsan``, try ``pip uninstall dmsan`` first.


or other fork and/or branch (replace ``<USERNAME_OF_THE_FORK>`` and ``<BRANCH_NAME>`` with the desired fork and branch names)

.. code::

    pip install git+https://github.com/<USERNAME_OF_THE_FORK>/DMsan.git@<BRANCH_NAME>


You can also download the package from `PyPI <https://pypi.org/project/dmsan/>`_.

Note that development of this package is currently under initial stage with limited backward compatibility, please feel free to `submit an issue <https://github.com/QSD-Group/DMsan/issues>`_ for any questions regarding package upgrading.

If you are a developer and want to contribute to ``QSDsan``, please follow the steps in the `Contributing to QSDsan <https://qsdsan.readthedocs.io/en/latest/CONTRIBUTING.html>`_ section of the documentation to clone the repository. If you find yourself struggle with the installation of QSDsan/setting up the environment, this extended version of `installation instructions <https://qsdsan.readthedocs.io/en/latest/tutorials/_installation.html>`_ might be helpful to you.


Documentation
-------------
You can find QSDsan tutorials and documents at:

   https://qsdsan.readthedocs.io

All tutorials are written using Jupyter Notebook, you can run your own Jupyter environment.

For each of these tutorials, we are also recording videos where one of the QSD group members will go through the tutorial step-by-step. We are gradually releasing these videos on our `YouTube channel <https://www.youtube.com/@qsd-group>`_ so subscribe to receive updates!


About the authors
-----------------
Development and maintenance of the package is supported by the Quantitative Sustainable Design Group led by members of the `Guest Group <http://engineeringforsustainability.com/>`_ at the `University of Illinois Urbana-Champaign (UIUC) <https://illinois.edu/>`_. Core contributors are listed below, please refer to the `author page <https://qsdsan.readthedocs.io/en/latest/AUTHORS.html>`_ for the full list of authors.

**Lead developers:**
   - `Hannah Lohman <https://qsdsan.readthedocs.io/en/beta/authors/Hannah_Lohman.html>`_
   - `Tori Morgan <https://qsdsan.readthedocs.io/en/beta/authors/Tori_Morgan.html>`_
   - `Yalin Li`_ (current maintainer)
   - `Joy Zhang`_


**Project conception & funding support:**
   - `Jeremy Guest <mailto:jsguest@illinois.edu>`_


Contributing
------------
Please refer to the `Contributing to QSDsan <https://qsdsan.readthedocs.io/en/latest/CONTRIBUTING.html>`_ section of the documentation for instructions and guidelines.


Stay Connected
--------------
If you would like to receive news related to the QSDsan platform, you can subscribe to email updates using `this form <https://groups.webservices.illinois.edu/subscribe/154591>`_ (don't worry, you will be able to unsubscribe :)). Thank you in advance for your interest!


QSDsan Events
-------------
We will keep this `calendar <https://calendar.google.com/calendar/embed?src=ep1au561lj8knfumpcd2a7ml08%40group.calendar.google.com&ctz=America%2FChicago>`_ up-to-date as we organize more events (office hours, workshops, etc.), click on the events in the calendar to see the details (including meeting links).


License information
-------------------
Please refer to the ``LICENSE.txt`` for information on the terms & conditions for usage of this software, and a DISCLAIMER OF ALL WARRANTIES.


References
----------
.. [1] Li, Y.; Trimmer, J.T.; Hand, S.; Zhang, X.; Chambers, K.G.; Lohman, H.A.C.; Shi, R.; Byrne, D.M.; Cook, S.M.; Guest, J.S. Quantitative Sustainable Design (QSD): A Methodology for the Prioritization of Research, Development, and Deployment of Technologies. (Tutorial Review) Environ. Sci.: Water Res. Technol. 2022, 8 (11), 2439–2465. https://doi.org/10.1039/D2EW00431C.

.. [2] Li, Y.; Zhang, X.; Morgan, V.L.; Lohman, H.A.C.; Rowles, L.S.; Mittal, S.; Kogler, A.; Cusick, R.D.; Tarpeh, W.A.; Guest, J.S. QSDsan: An integrated platform for quantitative sustainable design of sanitation and resource recovery systems. Environ. Sci.: Water Res. Technol. 2022, 8 (10), 2289-2303. https://doi.org/10.1039/d2ew00455k.


.. Links
.. _Yalin Li: https://qsdsan.readthedocs.io/en/beta/authors/Yalin_Li.html
.. _Joy Zhang: https://qsdsan.readthedocs.io/en/beta/authors/Joy_Zhang.html
