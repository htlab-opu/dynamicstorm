# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='dynamicstorm',
    version='0.1.1',
    description='manipulate data which output from DynamicStudio',
    long_description=readme,
    author='Kenya Igarashi',
    author_email='kenyayanke223@gmail.com',
    install_requires=['numpy', 'pandas', 'matplotlib', 'tqdm'],
    url='https://github.com/fiftystorm36/dynamicstorm',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    test_suite='tests'
)
