from setuptools import setup
from setuptools import find_packages

setup(name='ctalearn',
      version='0.1',
      description='CTA deep learning code (experimental)',
      url='http://github.com/',
      license='MIT',
      packages=['ctalearn','ctalearn.models','ctalearn.scripts'],
      dependencies=[],
      dependency_links=['git+http://github.com/cta-observatory/ctapipe'],
      zip_safe=False)
