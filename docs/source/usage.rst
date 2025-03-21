=============
Package usage
=============

A high-level companion python package to the CTLearn package is available for easy usage of the CTLearn tools. The package is called `CTLearnManager <https://ctlearn-manager.readthedocs.io/en/latest/>`_.

Low-level usage of CTLearn tools
--------------------------------

This page provides a brief overview of how to use the CTLearn tools. 

Training tool
-------------

To train a model, use the `ctlearn-train-model` command. The following command will display all available options for training a CTLearn model:

.. code-block:: bash

    ctlearn-train-model --help-all

View training progress in real time with TensorBoard: 

.. code-block:: bash

   tensorboard --logdir=/path/to/my/model_dir

Prediction tools 
----------------

To predict with a trained model, use the `ctlearn-predict-mono-model` or `ctlearn-predict-stereo-model` command. The following command will display all available options for predicting with a CTLearn model:

.. code-block:: bash

    ctlearn-predict-mono-model --help-all
    ctlearn-predict-stereo-model --help-all

.. CAUTION:: This tool expects the input data to be produced
   via the `ctapipe` package. The output file with the predictions
   follows the `ctapipe` DL2 data format.

To predict on real observational data from the LST1 telescope, use the `ctlearn-predict-LST1` command. The following command will display all available options for predicting with a CTLearn model:

.. code-block:: bash

    ctlearn-predict-LST1 --help-all

.. CAUTION:: This tool expects the input data to be produced
   via the `cta-lstchain` package. The output file with the predictions
   follows the `ctapipe` DL2 data format.
