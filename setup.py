from setuptools import setup, find_packages
from os import path
from ctlearn.version import *

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='ctlearn',
      version=get_version_pypi(),
      description='Deep learning for analysis and classification of image data for Imaging Atmospheric Cherenkov Telescopes, especially the Cherenkov Telescope Array (CTA).',
      long_description=long_description,
      long_description_content_type='text/x-rst',
      url='https://github.com/ctlearn-project/ctlearn',
      license='BSD-3-Clause',
      packages=['ctlearn'],
      install_requires=[
      'dl1_data_handler==0.8.3',
      'matplotlib',
      'numpy',
      'pandas',
      'pip',
      'pyyaml',
      'scikit-learn'
      ],
      entry_points = {
        'console_scripts': ['ctlearn=ctlearn.run_model:main'],
      },
      include_package_data=True,
      dependencies=[],
      dependency_links=[],
      zip_safe=False)
