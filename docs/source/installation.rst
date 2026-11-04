Installation
============

User Installation
-----------------

.. tab-set::

    .. tab-item:: With Pixi

        Inside an existing Pixi project, add the released CTLearn package:

        .. code-block:: shell

            $ pixi add --pypi ctlearn

    .. tab-item:: With Pip

        .. code-block:: shell

            $ pip install ctlearn


Developer Setup
---------------

As a developer, clone the repository and install CTLearn in development mode.
We support two methods: *pip/venv* and `Pixi <https://pixi.sh/>`__. Pixi
provides the same project-managed environment used by CI.

.. tab-set::

    .. tab-item:: With Pixi

        Install Pixi if needed, then clone the repository and install the
        development environment:

        .. code-block:: shell

            $ git clone https://github.com/ctlearn-project/ctlearn.git
            $ cd ctlearn
            $ pixi install -e dev

        Useful commands from the repository directory:

        .. code-block:: shell

            $ pixi run -e dev test            # run the test suite
            $ pixi run -e test-py313 test     # run tests with Python 3.13
            $ pixi run -e test-py314 test     # run tests with Python 3.14
            $ pixi run -e dev make -C docs html
            $ pixi shell -e dev               # open a shell in the dev environment
            $ pixi clean                      # remove Pixi environments

        Pixi targets Linux x86_64 and Apple Silicon macOS.

    .. tab-item:: With Pip

        .. code-block:: shell

            $ git clone https://github.com/ctlearn-project/ctlearn.git
            $ cd ctlearn
            $ python -m venv .venv
            $ source .venv/bin/activate
            $ python -m pip install --upgrade pip
            $ python -m pip install -e ".[dev]"


Installing on the IT Cluster
----------------------------

For working on the IT cluster, create a conda environment with the required
build tools:

.. code-block:: bash

   mamba create -n ctlearn-it-cluster -c conda-forge python==3.12 h5py scipy llvmlite gcc_linux-64 gxx_linux-64 openblas gfortran_linux-64
   conda activate ctlearn-it-cluster
   export CC=$(which x86_64-conda-linux-gnu-gcc)
   export CXX=$(which x86_64-conda-linux-gnu-g++)
   export FC=$(which x86_64-conda-linux-gnu-gfortran)
   pip install ctlearn

Update ``LD_LIBRARY_PATH`` to include the necessary paths. For example, add the
following line to your ``.bashrc`` file:

.. code-block:: bash

   export LD_LIBRARY_PATH=/path/to/your/conda/envs/ctlearn-it-cluster/lib:/path/to/cudnn/lib:$LD_LIBRARY_PATH

Replace ``/path/to/your/conda/envs/ctlearn-it-cluster/lib`` with the path to
your conda environment and ``/path/to/cudnn/lib`` with the path to the system's
cuDNN libraries for CUDA 12.


Core Dependencies
-----------------

* python>=3.12
* torch>=2.4.0
* tensorflow>=2.18
* keras>=3.0
* astropy
* ctapipe>=0.29.0
* dl1_data_handler>=0.14.12
* numba
* numpy
* pandas
* pyyaml


Uninstall CTLearn
-----------------

To remove a conda environment in which CTLearn is installed:

.. code-block:: bash

   conda remove --name ctlearn --all

To remove a source checkout, delete its directory.
