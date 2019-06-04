from setuptools import setup, find_packages

setup(
    name='neubio',
    version='0.0.1',
    description='Neurobiology lab processing utilities',
    author='Andy',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'click',
        'numpy',
        'pandas',
    ]
)