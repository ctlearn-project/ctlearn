=========================
Installation instructions
=========================

Install a released version
--------------------------

Download and install `Anaconda <https://www.anaconda.com/download/>`_\ , or, for a minimal installation, `Miniconda <https://conda.io/miniconda.html>`_.

The following command will set up a conda virtual environment, add the
necessary package channels, and install CTLearn specified version and its dependencies:

.. code-block:: bash

   CTLEARN_VER=0.6.2
   wget https://raw.githubusercontent.com/ctlearn-project/ctlearn/v$CTLEARN_VER/environment.yml
   conda env create -n [ENVIRONMENT_NAME] -f environment.yml
   conda activate [ENVIRONMENT_NAME]
   pip install ctlearn==$CTLEARN_VER
   ctlearn -h


This should automatically install all dependencies (NOTE: this may take some time, as by default MKL is included as a dependency of NumPy and it is very large).


Installing with pip/setuptools from source for development
----------------------------------------------------------

Clone the CTLearn repository:

.. code-block:: bash

   cd </ctlearn/installation/path>
   git clone https://github.com/ctlearn-project/ctlearn.git

First, install Anaconda by following the instructions above. Create a new conda environment that includes all the dependencies for CTLearn:

.. code-block:: bash

   conda env create -f </installation/path>/ctlearn/environment.yml

Finally, install CTLearn into the new conda environment via pypi:

.. code-block:: bash

   conda activate ctlearn
   pip install ctlearn==0.6.2

or with pip from source:

.. code-block:: bash

   conda activate ctlearn

   cd <ctlearn/installation/path>/ctlearn
   pip install .

NOTE for developers: If you wish to fork/clone the repository and edit the code, install with ``pip -e``.

Dependencies
------------

* Python>=3.9
* TensorFlow>=2.9
* tf2onnx>=1.12
* ctapipe==0.16.0
* ctaplot
* DL1DataHandler==0.10.8
* numba>=0.56.2
* NumPy
* PyYAML
* pyirf>=0.7
* Pandas
* Libraries used only in plotting scripts (optional)

  * Matplotlib
  * Scikit-learn

Uninstall CTLearn
-----------------

Remove Anaconda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, remove the conda environment in which CTLearn is installed and all its dependencies:

.. code-block:: bash

   conda remove --name ctlearn --all

Remove CTLearn
~~~~~~~~~~~~~~

Next, completely remove CTLearn from your system:

.. code-block:: bash

   rm -rf </installation/path>/ctlearn
