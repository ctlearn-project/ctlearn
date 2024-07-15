from setuptools import setup, find_packages
from os import path

def getVersionFromFile():
    file = open(".github/versionBackup.txt").readlines()
    for line in file:
        for word in line.split():
            return word

here = path.abspath(path.dirname(__file__))

def getRequirements():
    return open("docs/requirements.txt").readlines()
        
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='ctlearn',
      version=getVersionFromFile(),
      author="CTLearn Team",
      author_email="d.nieto@ucm.es",
      description='Deep learning analysis framework for Imaging Atmospheric Cherenkov Telescopes, especially the Cherenkov Telescope Array (CTA) and the MAGIC telescopes.',
      long_description=long_description,
      long_description_content_type='text/x-rst',
      url='https://github.com/ctlearn-project/ctlearn',
      license='BSD-3-Clause',
      packages=['ctlearn'],
      install_requires=getRequirements(),
      entry_points = {
        'console_scripts': ['ctlearn=ctlearn.run_model:main',
                            'build_irf=ctlearn.build_irf:main'],
      },
      
      dependency_links=[],
      zip_safe=True)
