from setuptools import setup, find_packages
from os import path
from ctlearn.version import *

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='ctlearn',
      version=get_version_pypi(),
      author="CTLearn Team",
      author_email="d.nieto@ucm.es",
      description='Deep learning analysis framework for Imaging Atmospheric Cherenkov Telescopes, especially the Cherenkov Telescope Array (CTA) and the MAGIC telescopes.',
      long_description=long_description,
      long_description_content_type='text/x-rst',
      url='https://github.com/ctlearn-project/ctlearn',
      license='BSD-3-Clause',
      packages=['ctlearn'],
      entry_points = {
        'console_scripts': ['ctlearn=ctlearn.run_model:main',
                            'build_irf=ctlearn.build_irf:main'],
        'ctapipe_reco': ['CTLearnReconstructor=ctlearn.ctapipe_plugin:CTLearnReconstructor'],
      },
      include_package_data=True,
      dependencies=[],
      dependency_links=[],
      zip_safe=False)
