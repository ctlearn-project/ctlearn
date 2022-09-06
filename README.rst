
CTLearn: Deep Learning for IACT Event Reconstruction
====================================================


.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3342952.svg
   :target: https://doi.org/10.5281/zenodo.3342952
   :alt: DOI


.. image:: https://img.shields.io/pypi/v/ctlearn
    :target: https://pypi.org/project/ctlearn/
    :alt: Latest Release


.. image:: images/CTLearnTextCTinBox_WhiteBkgd.png
   :target: images/CTLearnTextCTinBox_WhiteBkgd.png
   :alt: CTLearn Logo


CTLearn is a package under active development to run deep learning models to analyze data from all major current and future arrays of Imaging Atmospheric Cherenkov Telescopes (IACTs). CTLearn can load data from `CTA <https://www.cta-observatory.org/>`_ (Cherenkov Telescope Array), `FACT <https://www.isdc.unige.ch/fact/>`_\ , `H.E.S.S. <https://www.mpi-hd.mpg.de/hfm/HESS/>`_\ , `MAGIC <https://magic.mpp.mpg.de/>`_\ , and `VERITAS <https://veritas.sao.arizona.edu/>`_ telescopes processed by `ctapipe <https://github.com/cta-observatory/ctapipe>`_ or `DL1DataHandler <https://github.com/cta-observatory/dl1-data-handler>`_.

* Code, feature requests, bug reports, pull requests: https://github.com/ctlearn-project/ctlearn
* Documentation: https://ctlearn.readthedocs.io
* License: BSD-3

Installation for users
----------------------

Download and install `Anaconda <https://www.anaconda.com/download/>`_\ , or, for a minimal installation, `Miniconda <https://conda.io/miniconda.html>`_.

The following command will set up a conda virtual environment, add the
necessary package channels, and install CTLearn specified version and its dependencies:

.. code-block:: bash

   CTLEARN_VER=0.6.1
   wget https://raw.githubusercontent.com/ctlearn-project/ctlearn/v$CTLEARN_VER/environment.yml
   conda env create -n [ENVIRONMENT_NAME] -f environment.yml
   conda activate [ENVIRONMENT_NAME]
   pip install ctlearn==$CTLEARN_VER
   ctlearn -h

This should automatically install all dependencies (NOTE: this may take some time, as by default MKL is included as a dependency of NumPy and it is very large).

See the documentation for further information like `installation instructions for developers <https://ctlearn.readthedocs.io/en/stable/installation.html#installing-with-pip-setuptools-from-source-for-development>`_, `package usage <https://ctlearn.readthedocs.io/en/stable/usage.html>`_, and `dependencies <https://ctlearn.readthedocs.io/en/stable/installation.html#dependencies>`_ among other topics.

Citing this software
--------------------

Please cite the corresponding version using the DOIs below if this software package is used to produce results for any publication:

.. |zendoi060| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6842323.svg
   :target: https://doi.org/10.5281/zenodo.6842323
.. |zendoi052| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5947837.svg
   :target: https://doi.org/10.5281/zenodo.5947837
.. |zendoi051| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5772815.svg
   :target: https://doi.org/10.5281/zenodo.5772815
.. |zendoi050| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4576196.svg
   :target: https://doi.org/10.5281/zenodo.4576196
.. |zendoi040| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3345947.svg
   :target: https://doi.org/10.5281/zenodo.3345947
.. |zendoi040l| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3342954.svg
   :target: https://doi.org/10.5281/zenodo.3342954
.. |zendoi031| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3342953.svg
   :target: https://doi.org/10.5281/zenodo.3342953

* 0.6.0 : |zendoi050|
* 0.5.2 : |zendoi050|
* 0.5.1 : |zendoi050|
* 0.5.0 : |zendoi050|
* 0.4.0 : |zendoi040|
* 0.4.0-legacy : |zendoi040l|
* 0.3.1 : |zendoi031|
