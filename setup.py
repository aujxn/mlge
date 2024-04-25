#!/usr/bin/env python

from setuptools import setup, find_packages, find_namespace_packages

setup(name="mlge",
      version="1.0",
      description="Multilevel Graph Explorer",
      packages=find_namespace_packages(),
      package_dir={' ': 'mlge'},
      install_requires=['numpy',
                        'scipy',
                        'pyvista'],
      python_requires='>=3.6'
      )
