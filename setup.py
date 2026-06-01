from setuptools import setup, find_packages

setup(
    name='data-analysis-tool',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'plotly',
        'scipy',
        'scikit-learn',
        'statsmodels',
        'pydantic'
    ]
)
