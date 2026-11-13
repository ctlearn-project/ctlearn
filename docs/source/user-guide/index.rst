.. _user-guide:

**********
User Guide
**********

.. currentmodule:: ctlearn


Getting Started for Users
-------------------------

.. tab-set::

    .. tab-item:: With Pixi

         Inside an existing Pixi project, add the released CTLearn package:

         .. code-block:: shell

            $ pixi add --pypi ctlearn

    .. tab-item:: With Pip

         .. code-block:: shell

            $ pip install ctlearn

    .. tab-item:: On IT-Cluster

         For working on the IT cluster, create a conda environment with the required
         build tools:

         .. code-block:: shell

            $ mamba create -n ctlearn-it-cluster -c conda-forge python==3.12 h5py scipy llvmlite gcc_linux-64 gxx_linux-64 openblas gfortran_linux-64
            $ conda activate ctlearn-it-cluster
            $ export CC=$(which x86_64-conda-linux-gnu-gcc)
            $ export CXX=$(which x86_64-conda-linux-gnu-g++)
            $ export FC=$(which x86_64-conda-linux-gnu-gfortran)
            $ pip install ctlearn

         Update ``LD_LIBRARY_PATH`` to include the necessary paths. For example, add the
         following line to your ``.bashrc`` file:

         .. code-block:: shell

            $ export LD_LIBRARY_PATH=/path/to/your/conda/envs/ctlearn-it-cluster/lib:/path/to/cudnn/lib:$LD_LIBRARY_PATH

         Replace ``/path/to/your/conda/envs/ctlearn-it-cluster/lib`` with the path to
         your conda environment and ``/path/to/cudnn/lib`` with the path to the system's
         cuDNN libraries for CUDA 12.

.. toctree::
   :maxdepth: 1

   auto_tutorials/index

.. toctree::
   :hidden:

   usage
   supplementary
