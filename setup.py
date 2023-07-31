#! /usr/bin/env python
# flake8: noqa
from setuptools import find_packages, setup

setup(
    name="mks-test",
    version="1.0.0",
    description="Python implementation of an extension of the Kolmogorov-Smirnov test for multivariate samples",
    author="Olivier Laurent",
    author_email="olivier.laurent@ensta-paris.fr",
    install_requires=[
        "numpy",
    ],
    packages=find_packages(),
)
