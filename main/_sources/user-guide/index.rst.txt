.. _user-guide:

**********
User Guide
**********

.. currentmodule:: ctlearn


Getting Started for Users
-------------------------

.. warning::

   The following guide is for *users*. If you want to contribute to
   CTLearn as a developer, see :ref:`dev-guide`.


Installation for Users
======================

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


Basic package usage
===================

The CTLearn package provides a set of command-line tools for training and predicting with deep learning models
using Keras or PyTorch. The tools are designed to work with data produced by the ``ctapipe`` package following
the CTAO data format. However, it also supports predicting on real observational data from the LST1 telescope
using the ``cta-lstchain`` package for the time being. Detail tutorials on how to use the CTLearn tools
are available in the :ref:`ctlearn-tutorials` section of this documentation. Additionally, a high-level companion
python package to the CTLearn package is available for easy usage of the CTLearn tools.
The package is called `CTLearnManager <https://ctlearn-manager.readthedocs.io/en/latest/>`_.

Training tools
^^^^^^^^^^^^^^

To train a model, choose the Keras or PyTorch training tool. Use ``--help-all``
to display its available options. The examples below train a ResNet model for
particle classification:

.. tab-set::

    .. tab-item:: Keras

        .. code-block:: shell

            $ ctlearn-train-keras-model --help-all
            $ ctlearn-train-keras-model \
                --signal=/path/to/signal_dir \
                --pattern-signal="*gamma*.dl1.h5" \
                --background=/path/to/background_dir \
                --pattern-background="*proton*.dl1.h5" \
                --config=/path/to/reader_config.json \
                --output=/path/to/output_dir_keras \
                --reco=type \
                --TrainCTLearnModel.model_type=ResNet \
                --TrainCTLearnModel.n_epochs=10 \
                --TrainCTLearnModel.batch_size=64

    .. tab-item:: PyTorch

        .. code-block:: shell

            $ ctlearn-train-pytorch-model --help-all
            $ ctlearn-train-pytorch-model \
                --signal=/path/to/signal_dir \
                --pattern-signal="*gamma*.dl1.h5" \
                --background=/path/to/background_dir \
                --pattern-background="*proton*.dl1.h5" \
                --config=/path/to/reader_config.json \
                --output=/path/to/output_dir_pytorch \
                --reco=type \
                --TrainCTLearnModel.model_type=ResNet \
                --TrainCTLearnModel.n_epochs=10 \
                --TrainCTLearnModel.batch_size=64

View training progress in real time with TensorBoard:

.. code-block:: shell

   $ tensorboard --logdir=/path/to/output_dir

Prediction tools
^^^^^^^^^^^^^^^^

To predict with a trained Keras or PyTorch model, use the ``ctlearn-predict-mono-model`` or ``ctlearn-predict-stereo-model`` command.
The following command will display all available options for predicting with a CTLearn model:

.. code-block:: shell

    $ ctlearn-predict-mono-model --help-all
    $ ctlearn-predict-stereo-model --help-all

**Example: Predicting with a trained mono classification model**

Use the same reader configuration as for training:

.. tab-set::

    .. tab-item:: Keras

        .. code-block:: shell

            $ ctlearn-predict-mono-model \
                --input_url=/path/to/input.dl1.h5 \
                --output=/path/to/keras_prediction_output.dl2.h5 \
                --config=/path/to/reader_config.json \
                --type_model=/path/to/output_dir_keras/ctlearn_model.keras \
                --PredictCTLearnModel.batch_size=64

    .. tab-item:: PyTorch

        .. code-block:: shell

            $ ctlearn-predict-mono-model \
                --input_url=/path/to/input.dl1.h5 \
                --output=/path/to/pytorch_prediction_output.dl2.h5 \
                --config=/path/to/reader_config.json \
                --type_model=/path/to/output_dir_pytorch/ctlearn_model.pth \
                --PredictCTLearnModel.batch_size=64

.. CAUTION:: This tool expects the input data to be produced
   via the `ctapipe` package. The output file with the predictions
   follows the `ctapipe` DL2 data format.

View the predictions in a GUI with `ViTables <https://vitables.org/>`_:

.. code-block:: shell

   $ pip install vitables
   $ vitables /path/to/*_prediction_output.dl2.h5



LST-1 prediction tool
^^^^^^^^^^^^^^^^^^^^^

To predict on real observational data from the LST1 telescope, use the ``ctlearn-predict-LST1`` command.
The following command will display all available options for predicting with a CTLearn model:

.. code-block:: shell

    $ ctlearn-predict-LST1 --help-all

**Example: Predicting on LST1 observational data**

Use the same image channels and mapping as for training:

.. tab-set::

    .. tab-item:: Keras

        .. code-block:: shell

            $ ctlearn-predict-LST1 \
                --input_url=/path/to/lst1_input.dl1.h5 \
                --output=/path/to/lst1_keras_prediction_output.dl2.h5 \
                --type_model=/path/to/type/ctlearn_model.keras \
                --energy_model=/path/to/energy/ctlearn_model.keras \
                --LST1PredictionTool.channels=cleaned_image \
                --LST1PredictionTool.channels=cleaned_relative_peak_time \
                --LST1PredictionTool.image_mapper_type=BilinearMapper

    .. tab-item:: PyTorch

        .. code-block:: shell

            $ ctlearn-predict-LST1 \
                --input_url=/path/to/lst1_input.dl1.h5 \
                --output=/path/to/lst1_pytorch_prediction_output.dl2.h5 \
                --type_model=/path/to/type/ctlearn_model.pth \
                --energy_model=/path/to/energy/ctlearn_model.pth \
                --LST1PredictionTool.channels=cleaned_image \
                --LST1PredictionTool.channels=cleaned_relative_peak_time \
                --LST1PredictionTool.image_mapper_type=BilinearMapper

.. CAUTION:: This tool expects the input data to be produced
   via the ``cta-lstchain`` package. The output file with the predictions
   follows the ``ctapipe`` DL2 data format.

For step-by-step examples from the different workflows, see the :doc:`CTLearn tutorials <auto_tutorials/index>`.

.. toctree::
   :hidden:

   auto_tutorials/index
   supplementary
