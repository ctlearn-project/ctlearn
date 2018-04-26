from setuptools import setup
from setuptools import find_packages

setup(name='ctalearn',
      version='0.1.1',
      description='Deep learning models for analysis and classification of image data for CTA (the Cherenkov Telescope Array).',
      url='https://github.com/bryankim96/ctalearn',
      license='MIT',
      packages=['ctalearn','ctalearn.scripts'],
      dependencies=[],
      dependency_links=[],
      zip_safe=False)
