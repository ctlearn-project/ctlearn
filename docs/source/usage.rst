=============
Package usage
=============

A high-level companion python package to the CTLearn package is available for easy usage of the CTLearn tools. The package is called `CTLearnManager <https://ctlearn-manager.readthedocs.io/en/latest/>`_.

Low-level usage of CTLearn tools
--------------------------------

This page provides a brief overview of how to use the CTLearn tools. 

Training tool
-------------

To train a model, use the `ctlearn-train-keras-model` or `ctlearn-train-pytorch-model` command. The following command will display all available options for training a CTLearn model:

.. code-block:: bash

    ctlearn-train-keras-model --help-all
    ctlearn-train-pytorch-model --help-all

**Example: Training a PyTorch ResNet model for particle classification**

.. code-block:: bash

    ctlearn-train-pytorch-model \
        --signal=/path/to/signal_dir \
        --pattern-signal="*gamma*.dl1.h5" \
        --background=/path/to/background_dir \
        --pattern-background="*proton*.dl1.h5" \
        --output=/path/to/output_dir \
        --reco=type \
        --TrainCTLearnModel.model_type=ResNet \
        --TrainCTLearnModel.n_epochs=10 \
        --TrainCTLearnModel.batch_size=64

You can use the exact same arguments with `ctlearn-train-keras-model` to train using the Keras framework.

View training progress in real time with TensorBoard: 

.. code-block:: bash

   tensorboard --logdir=/path/to/output_dir

Prediction tools 
----------------

To predict with a trained Keras or PyTorch model, use the `ctlearn-predict-mono-model` or `ctlearn-predict-stereo-model` command. The following command will display all available options for predicting with a CTLearn model:

.. code-block:: bash

    ctlearn-predict-mono-model --help-all
    ctlearn-predict-stereo-model --help-all

**Example: Predicting with a trained stereo model**

.. code-block:: bash

    ctlearn-predict-stereo-model \
        --input=/path/to/data_dir \
        --pattern="*.dl1.h5" \
        --output=/path/to/prediction_output_dir \
        --PredictCTLearnModel.load_model_from=/path/to/output_dir/ctlearn_model.pt \
        --PredictCTLearnModel.reco_tasks="['type']" \
        --PredictCTLearnModel.batch_size=64

.. CAUTION:: This tool expects the input data to be produced
   via the `ctapipe` package. The output file with the predictions
   follows the `ctapipe` DL2 data format.

To predict on real observational data from the LST1 telescope, use the `ctlearn-predict-LST1` command. The following command will display all available options for predicting with a CTLearn model:

.. code-block:: bash

    ctlearn-predict-LST1 --help-all

**Example: Predicting on LST1 observational data**

.. code-block:: bash

    ctlearn-predict-LST1 \
        --input=/path/to/lst1_data_dir \
        --pattern="Run*.h5" \
        --output=/path/to/lst1_prediction_output_dir \
        --PredictCTLearnModel.load_model_from=/path/to/output_dir/ctlearn_model.keras \
        --PredictCTLearnModel.reco_tasks="['type', 'energy']"

.. CAUTION:: This tool expects the input data to be produced
   via the `cta-lstchain` package. The output file with the predictions
   follows the `ctapipe` DL2 data format.
