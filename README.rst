
CTLearn: Deep Learning for IACT Event Reconstruction
====================================================


.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3342952.svg
   :target: https://doi.org/10.5281/zenodo.3342952
   :alt: DOI


.. image:: https://travis-ci.com/ctlearn-project/ctlearn.svg?branch=master
   :target: https://travis-ci.com/ctlearn-project/ctlearn
   :alt: Build Status

.. image:: https://img.shields.io/pypi/v/ctlearn
    :target: https://pypi.org/project/ctlearn/
    :alt: Latest Release


.. image:: images/CTLearnTextCTinBox_WhiteBkgd.png
   :target: images/CTLearnTextCTinBox_WhiteBkgd.png
   :alt: CTLearn Logo


CTLearn is a package under active development to run deep learning models to analyze data from all major current and future arrays of Imaging Atmospheric Cherenkov Telescopes (IACTs). CTLearn can load data from `CTA <https://www.cta-observatory.org/>`_ (Cherenkov Telescope Array), `FACT <https://www.isdc.unige.ch/fact/>`_\ , `H.E.S.S. <https://www.mpi-hd.mpg.de/hfm/HESS/>`_\ , `MAGIC <https://magic.mpp.mpg.de/>`_\ , and `VERITAS <https://veritas.sao.arizona.edu/>`_ telescopes processed using `DL1DataHandler <https://github.com/cta-observatory/dl1-data-handler>`_.

Install CTLearn
---------------

Install a released version
^^^^^^^^^^^^^^^^^^^^^^^^^^

Download and install `Anaconda <https://www.anaconda.com/download/>`_\ , or, for a minimal installation, `Miniconda <https://conda.io/miniconda.html>`_.

The following command will set up a conda virtual environment, add the
necessary package channels, and install CTLearn specified version and its dependencies:

.. code-block:: bash

   CTLEARN_VER=0.6.0
   mode=cpu
   wget https://raw.githubusercontent.com/ctlearn-project/ctlearn/v$CTLEARN_VER/environment-$mode.yml
   conda env create -n [ENVIRONMENT_NAME] -f environment-$mode.yml
   conda activate [ENVIRONMENT_NAME]
   pip install ctlearn=$CTLEARN_VER
   ctlearn -h

where ``mode`` is either 'cpu' or 'gpu' (for linux systems) or 'macos' (for macOS systems), denoting the TensorFlow version to be installed. If installing the GPU version of TensorFlow, verify that your system fulfills all the requirements `here <https://www.tensorflow.org/install/install_linux#NVIDIARequirements>`_. Note that there is no GPU-enabled TensorFlow version for macOS yet.

This should automatically install all dependencies (NOTE: this may take some time, as by default MKL is included as a dependency of NumPy and it is very large).


Installing with pip/setuptools from source for development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the CTLearn repository:

.. code-block:: bash

   cd </ctlearn/installation/path>
   git clone https://github.com/ctlearn-project/ctlearn.git

First, install Anaconda by following the instructions above. Create a new conda environment that includes all the dependencies for CTLearn:

.. code-block:: bash

   conda env create -f </installation/path>/ctlearn/environment-<MODE>.yml

where ``<MODE>`` is either 'cpu' or 'gpu' (for linux systems) or 'macos' (for macOS systems), denoting the TensorFlow version to be installed (see above).

Finally, install CTLearn into the new conda environment via pypi:

.. code-block:: bash

   conda activate ctlearn
   pip install ctlearn==0.6.0

or with pip from source:

.. code-block:: bash

   conda activate ctlearn

   cd <ctlearn/installation/path>/ctlearn
   pip install .

NOTE for developers: If you wish to fork/clone the repository and edit the code, install with ``pip -e``.

Dependencies
^^^^^^^^^^^^


* Python>=3.8
* TensorFlow>=2.8
* ctapipe==0.12.0
* DL1DataHandler==0.10.5
* NumPy
* PyYAML
* Pandas
* Libraries used only in plotting scripts (optional)

  * Matplotlib
  * Scikit-learn

Download Data
-------------

CTLearn can load and process data in the HDF5 PyTables format produced from simtel files by `ctapipe <https://github.com/cta-observatory/ctapipe>`_ and `DL1DataHandler <https://github.com/cta-observatory/dl1-data-handler>`_.

Configure a Run
---------------

CTLearn encourages reproducible training and prediction by keeping all run settings in a single YAML configuration file, organized into the sections listed below. The `example config file <config/example_config.yml>`_ describes every available setting and its possible values in detail. Predefined default CTLearn models are shipped with the installation and can be constructed via ``--default_model,-d`` from the command line.

Logging
^^^^^^^

Specify model directory to store TensorFlow checkpoints and summaries, a timestamped copy of the run configuration, and optionally a timestamped file with logging output.

Data
^^^^

Describe the dataset to use and relevant settings for loading and processing it. The parameters in this section are used to initialize a DL1DataReader, which loads the data files, maps the images from vectors to arrays, applies preprocessing, and returns the data as an iterator. Data can be loaded in two modes:


* Mono: single images of one telescope type
* Stereo: events of one or multiple telescope types

Parameters in this section include telescope IDs to select, auxiliary parameters to return, pre-selection cuts, image mapping settings, and pre-processing to apply to the data. Image mapping is performed by the DL1DataReader and maps the 1D pixel vectors in the raw data into 2D images. The available mapping methods are oversampling, nearest interpolation, rebinning, bilinear interpolation and bicubic interpolation, image shifting, and axial addressing.
Pre-processing is performed using the DL1DataHandler Transform class.

Input
^^^^^

Set parameters of the KerasBatchGenerator that converts the loaded, processed data into generator of batches for the Keras application. Stereoscopic images can be concatenated via the ``concat_telescopes`` flag.

Model
^^^^^

CTLearn works with any TensorFlow-Keras model obeying the signature of a backbone_model (``backbone, backbone_inputs = backbone_model(data, model_params)`` where ``backbone`` is a TensorFlow-Keras (sub)model with model inputs ``backbone_inputs``, ``data`` is a KerasBatchGenerator, and ``model_params`` is a dictionary of model parameters) and a head_model (``logits, losses, loss_weights, metrics = head_model(backbone_output, tasks, model_params)`` where ``backbone_output`` is an output of a TensorFlow-Keras backbone model, ``tasks`` is a list of reconstruction tasks, ``model_params`` is a dictionary of model parameters, and ``logits``, ``losses``, ``loss_weights``, ``metrics`` are lists of self-explanatory outputs correspondent to the selected tasks).

To use a custom model, provide in this section the directory containing a Python file that implements the model and the module name (that is, the file name minus the .py extension) and name of the model function within the module.

In addition, CTLearn includes two main models for gamma/hadron classification, energy and arrival direction regression. ``SingleCNN`` analyzes single telescope images using a convolutional neural network (CNN) or multiple residual blocks of convolutional layers (ResNet). Stereoscopic images can be concatenated beforehand (in the ``Input`` config section) to be analyzed by the ``SingleCNN`` model. ``CNN-RNN`` performs array-level reconstruction by feeding the output of a CNN or a ResNet for each telescope into either a recurrent neural network (RNN). All models are built on generic functions from ``basic.py`` and ``resnet_engine.py``. In addition, three different attention mechanisms are implemented in ``attention.py``.

Model Parameters
^^^^^^^^^^^^^^^^

This section in its entirety is directly included as the model ``params``\ , enabling arbitrary configuration parameters to be passed to the provided model.

Training
^^^^^^^^

Set training parameters such as the training/validation split, the number of epochs to run, as well as hyperparameters including the base learning rate and optimizer.

Prediction
^^^^^^^^^^

Specify prediction settings such as the path to write the prediction file and whether to save the labels and example identifiers along with the predictions.

TensorFlow
^^^^^^^^^^

Set whether to run TensorFlow in debug mode.

Run a Model
-----------

Run CTLearn from the command line:

.. code-block:: bash

   ctlearn [-h] [--config_file,-c CONFIG_FILE] [--input,-i INPUT] [--pattern,-p PATTERN [PATTERN ...]] [--mode,-m MODE] [--output,-o OUTPUT] [--reco,-r RECO [RECO ...]]
                [--default_model,-d DEFAULT_MODEL] [--pretrained_weights,-w PRETRAINED_WEIGHTS] [--tel_types,-t TEL_TYPES [TEL_TYPES ...]] [--allowed_tels,-a ALLOWED_TELS [ALLOWED_TELS ...]]
                [--size_cut,-z SIZE_CUT] [--leakage_cut,-l LEAKAGE_CUT] [--multiplicity_cut,-u MULTIPLICITY_CUT] [--num_epochs,-e NUM_EPOCHS] [--batch_size,-b BATCH_SIZE] [--random_seed,-s RANDOM_SEED]
                [--log_to_file] [--debug]

optional arguments:
  ``-h, --help``\
                        show this help message and exit
  ``--config_file,-c CONFIG_FILE``\
                        Path to YAML configuration file with training options
  ``--input,-i INPUT``\
                        Input directory (not required when file_list is set in the config file)
  ``--pattern,-p PATTERN [PATTERN ...]``\
                        Pattern to mask unwanted files from the data input directory
  ``--mode,-m MODE``\
                        Mode to run CTLearn; valid options: train, predict, or train_and_predict
  ``--output,-o OUTPUT``\
                        Output directory, where the logging, model weights and processed output files are stored
  ``--reco,-r RECO [RECO ...]``\
                        Reconstruction task to perform; valid options: particletype, energy, and/or direction
  ``--default_model,-d DEFAULT_MODEL``\
                        Default CTLearn Model; valid options: TRN, TRN_cleaned, mergedTRN, mergedTRN_cleaned, CNNRNN, and CNNRNN_cleaned
  ``--pretrained_weights,-w PRETRAINED_WEIGHTS``\
                        Path to the pretrained weights
  ``--tel_types,-t TEL_TYPES [TEL_TYPES ...]``\
                        Selection of telescope types; valid option: LST_LST_LSTCam, LST_MAGIC_MAGICCam, MST_MST_FlashCam, MST_MST_NectarCam, SST_SCT_SCTCam, and/or SST_ASTRI_ASTRICam
  ``--allowed_tels,-a ALLOWED_TELS [ALLOWED_TELS ...]``\
                        List of allowed tel_ids, others will be ignored. Selected tel_ids will be ignored, when their telescope type is not selected
  ``--size_cut,-z SIZE_CUT``\
                        Hillas intensity cut to perform
  ``--leakage_cut,-l LEAKAGE_CUT``\
                        Leakage intensity cut to perform
  ``--multiplicity_cut,-u MULTIPLICITY_CUT``\
                        Multiplicity cut to perform
  ``--num_epochs,-e NUM_EPOCHS``\
                        Number of epochs to train
  ``--batch_size,-b BATCH_SIZE``\
                        Batch size per worker
  ``--random_seed,-s RANDOM_SEED``\
                        Selection of random seed (4 digits)
  ``--log_to_file``\
                        Log to a file in model directory instead of terminal
  ``--debug``\
                        Print debug/logger messages

Alternatively, import CTLearn as a module in a Python script:

.. code-block:: python

   import yaml
   from ctlearn.run_model import run_model

   with open('myconfig.yml', 'r') as myconfig:
     config = yaml.load(myconfig)
   run_model(config, mode='train', debug=True, log_to_file=True)

View training progress in real time with TensorBoard: 

.. code-block:: bash

   tensorboard --logdir=/path/to/my/model_dir


Supplementary Scripts
---------------------

* **plot_classifier_values.py** Plot a histogram of gamma/hadron classification values from a CTLearn predictions file.
* **plot_roc_curves.py** Plot gamma/hadron classification ROC curves from a list of CTLearn predictions files.
* [Deprecated] **run_multiple_configurations.py** Generate a list of configuration combinations and run a model for each, for example, to conduct a hyperparameter search or to automate training or prediction for a set of models. Parses a standard CTLearn configuration file with two additional sections for Multiple Configurations added. Has an option to resume from a specific run in case the execution is interrupted.
* [Deprecated] **auto_configuration.py** Fill the path information specific to your computer and run this script from a folder with any number of configuration files to automatically overwrite them.
* [Deprecated] **summarize_results.py** Run this script from the folder containing the ``runXX`` folders generated by the ``run_multiple_configurations.py`` script to generate a ``summary.csv`` file with key validation metrics after training of each run.

Uninstall CTLearn
-----------------

Remove Anaconda Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, remove the conda environment in which CTLearn is installed and all its dependencies:

.. code-block:: bash

   conda remove --name ctlearn --all

Remove CTLearn
^^^^^^^^^^^^^^

Next, completely remove CTLearn from your system:

.. code-block:: bash

   rm -rf </installation/path>/ctlearn
