from setuptools import setup, find_packages

setup(name='ctlearn',
      version='0.5.0',
      description='Deep learning for analysis and classification of image data for Imaging Atmospheric Cherenkov Telescopes, especially the Cherenkov Telescope Array (CTA).',
      url='https://github.com/ctlearn-project/ctlearn',
      license='BSD-3-Clause',
      packages=['ctlearn'],
      entry_points = {
        'console_scripts': ['ctlearn=ctlearn.run_model:main'],
      },
      include_package_data=True,
      dependencies=[],
      dependency_links=[],
      zip_safe=False)
