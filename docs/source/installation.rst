=========================
Installation instructions
=========================

Install a released version
--------------------------

Download and install `Anaconda <https://www.anaconda.com/download/>`_\ , or, for a minimal installation, `Miniconda <https://conda.io/miniconda.html>`_.

The following command will set up a conda virtual environment, add the
necessary package channels, and install CTLearn specified version and its dependencies:

.. code-block:: bash

   mamba create -n ctlearn -c conda-forge python==3.12 llvmlite
   conda activate ctlearn
   pip install ctlearn

For working on the IT-cluster:

.. code-block:: bash

   mamba create -n ctlearn-it-cluster -c conda-forge python==3.12 h5py scipy llvmlite gcc_linux-64 gxx_linux-64 openblas gfortran_linux-64
   conda activate ctlearn-it-cluster
   export CC=$(which x86_64-conda-linux-gnu-gcc)
   export CXX=$(which x86_64-conda-linux-gnu-g++)
   export FC=$(which x86_64-conda-linux-gnu-gfortran)
   pip install ctlearn

Please do not forget to update your LD_LIBRARY_PATH to include the necessary paths. For example, you can add the following line to your .bashrc file:
export LD_LIBRARY_PATH=/to/your/.conda/envs/ctlearn-it-cluster/lib:/fefs/aswg/workspace/tjark.miener/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib:$LD_LIBRARY_PATH
Note: You would need to replace the /to/your/.conda/envs/ctlearn-it-cluster/lib with the path to your conda environment where ctlearn is installed. cudnn-linux-x86_64-8.9.7.29_cuda12-archive is the path to the cuDNN libraries for CUDA 12.


Installing with pip/setuptools from source for development
----------------------------------------------------------

First, install Anaconda by following the instructions above. Create a new conda environment that includes all the dependencies for CTLearn. Then, clone the CTLearn repository and install CTLearn into the new conda environment with pip from source:

.. code-block:: bash

   mamba create -n ctlearn -c conda-forge python==3.12 llvmlite
   conda activate ctlearn
   cd </ctlearn/installation/path>
   git clone https://github.com/ctlearn-project/ctlearn.git
   pip install -e .


Core dependencies
-----------------

* python>=3.12
* torch>=2.4.0
* tensorflow>=2.16
* keras>=3.0
* ctapipe>=0.29.0
* ctaplot
* dl1_data_handler>=0.14.10
* numba
* numpy
* pandas
* pyyaml

* Libraries used only in plotting scripts (optional)

  * Matplotlib
  * Scikit-learn
  * ctaplot
  * pyirf

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
