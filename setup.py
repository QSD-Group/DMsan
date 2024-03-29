#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DMsan: Decision-making for sanitation and resource recovery systems

This module is developed by:

    Yalin Li <mailto.yalin.li@gmail.com>

This module is under the University of Illinois/NCSA Open Source License.
Please refer to https://github.com/QSD-Group/DMsan/blob/main/LICENSE.txt
for license details.
'''

from setuptools import setup

setup(
    name='dmsan',
    packages=['dmsan'],
    version='0.1.0',
    license='UIUC',
    author='Quantitative Sustainable Design Group',
    author_email='quantitative.sustainable.design@gmail.com',
    description='Decision-Making for sanitation and resource recovery systems',
    long_description=open('README.rst', encoding='utf-8').read(),
    url="https://github.com/QSD-Group/DMsan",
    install_requires=['exposan', 'country-converter'],
    package_data={
        'dmsan': [
           'biogenic_refinery/*',
           'bwaise/*',
           'comparison/*',
           'comparison/data/*',
           'data/*',
           'new_generator/*',
           'reclaimer/*',
    ]},
    classifiers=[
        'License :: OSI Approved :: University of Illinois/NCSA Open Source License',
        'Environment :: Console',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Manufacturing',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords=[
        'multi-criteria decision analysis',
        'quantitative sustainable design',
        'sanitation', 'resource recovery'
    ],
)