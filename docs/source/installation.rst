=========================
Installation instructions
=========================

Install a released version
--------------------------

Download and install `Anaconda <https://www.anaconda.com/download/>`_\ , or, for a minimal installation, `Miniconda <https://conda.io/miniconda.html>`_.

The following command will set up a conda virtual environment, add the
necessary package channels, and install CTLearn specified version and its dependencies:

.. code-block:: bash

   # Linux:
   mamba create -n ctlearn -c conda-forge python==3.12 llvmlite triton
   # macOS/Windows:
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

Please do not forget to update your ``LD_LIBRARY_PATH`` to include the necessary paths. For example, you can add the following line to your ``.bashrc`` file:

.. code-block:: bash

   export LD_LIBRARY_PATH=/path/to/your/conda/envs/ctlearn-it-cluster/lib:/path/to/cudnn/lib:$LD_LIBRARY_PATH

.. note:: 
   You will need to replace ``/path/to/your/conda/envs/ctlearn-it-cluster/lib`` with the actual path to your conda environment where CTLearn is installed. Similarly, replace ``/path/to/cudnn/lib`` with the path to your system's cuDNN libraries for CUDA 12.


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
* astropy
* ctapipe>=0.29.0
* dl1_data_handler>=0.14.10
* numba
* numpy
* pandas
* pyyaml

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
