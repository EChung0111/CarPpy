from setuptools import setup, find_packages

setup(
    name='carp',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'networkx'
    ],
)
