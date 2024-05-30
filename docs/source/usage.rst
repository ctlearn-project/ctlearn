=============
Package usage
=============

Download data
-------------

CTLearn can load and process data in the HDF5 PyTables format produced from simtel files by `ctapipe <https://github.com/cta-observatory/ctapipe>`_ and `DL1DataHandler <https://github.com/cta-observatory/dl1-data-handler>`_.

Configure a run
---------------

CTLearn encourages reproducible training and prediction by keeping all run settings in a single YAML configuration file, organized into the sections listed below. The `example config file <config/example_config.yml>`_ describes every available setting and its possible values in detail. Predefined default CTLearn models are shipped with the installation and can be constructed via ``--default_model,-d`` from the command line.

Logging
~~~~~~~

Specify model directory to store TensorFlow checkpoints and summaries, a timestamped copy of the run configuration, and optionally a timestamped file with logging output.

Data
~~~~

Describe the dataset to use and relevant settings for loading and processing it. The parameters in this section are used to initialize a DL1DataReader, which loads the data files, maps the images from vectors to arrays, applies preprocessing, and returns the data as an iterator. Data can be loaded in two modes:

* Mono: single images of one telescope type
* Stereo: events of one or multiple telescope types

Parameters in this section include telescope IDs to select, auxiliary parameters to return, pre-selection cuts, image mapping settings, and pre-processing to apply to the data. Image mapping is performed by the DL1DataReader and maps the 1D pixel vectors in the raw data into 2D images. The available mapping methods are oversampling, nearest interpolation, rebinning, bilinear interpolation and bicubic interpolation, image shifting, and axial addressing.
Pre-processing is performed using the DL1DataHandler Transform class.

Input
~~~~~

Set parameters of the KerasBatchGenerator that converts the loaded, processed data into generator of batches for the Keras application. Stereoscopic images can be stacked via the ``stack_telescope_images`` flag.

Model
~~~~~

CTLearn works with any TensorFlow-Keras model obeying the signature of a backbone_model (``backbone, backbone_inputs = backbone_model(data, model_params)`` where ``backbone`` is a TensorFlow-Keras (sub)model with model inputs ``backbone_inputs``, ``data`` is a KerasBatchGenerator, and ``model_params`` is a dictionary of model parameters) and a head_model (``logits, losses, loss_weights, metrics = head_model(backbone_output, tasks, model_params)`` where ``backbone_output`` is an output of a TensorFlow-Keras backbone model, ``tasks`` is a list of reconstruction tasks, ``model_params`` is a dictionary of model parameters, and ``logits``, ``losses``, ``loss_weights``, ``metrics`` are lists of self-explanatory outputs correspondent to the selected tasks).

To use a custom model, provide in this section the directory containing a Python file that implements the model and the module name (that is, the file name minus the .py extension) and name of the model function within the module.

In addition, CTLearn includes two main models for gamma/hadron classification, energy and arrival direction regression. ``SingleCNN`` analyzes single telescope images using a convolutional neural network (CNN) or multiple residual blocks of convolutional layers (ResNet). Stereoscopic images can be stacked beforehand (in the ``Input`` config section) to be analyzed by the ``SingleCNN`` model. ``CNN-RNN`` performs array-level reconstruction by feeding the output of a CNN or a ResNet for each telescope into either a recurrent neural network (RNN). All models are built on generic functions from ``basic.py`` and ``resnet_engine.py``. In addition, three different attention mechanisms are implemented in ``attention.py``.

Model Parameters
~~~~~~~~~~~~~~~~

This section in its entirety is directly included as the model ``params``\ , enabling arbitrary configuration parameters to be passed to the provided model.

Training
~~~~~~~~

Set training parameters such as the training/validation split, the number of epochs to run, as well as hyperparameters including the base learning rate and optimizer.

Prediction
~~~~~~~~~~

Specify prediction settings such as the path to write the prediction file and whether to save the labels and example identifiers along with the predictions.

TensorFlow
~~~~~~~~~~

Set whether to run TensorFlow in debug mode.

Run a model
-----------

Run CTLearn from the command line:

.. code-block:: bash

   ctlearn [-h] [--config_file,-c CONFIG_FILE] [--input,-i INPUT] [--pattern,-p PATTERN [PATTERN ...]] [--mode,-m MODE] [--output,-o OUTPUT] [--reco,-r RECO [RECO ...]]
                [--default_model,-d DEFAULT_MODEL] [--clean | --no-clean] [--pretrained_weights,-w PRETRAINED_WEIGHTS] [--prediction_directory,-y PREDICTION_DIRECTORY] [--tel_types,-t TEL_TYPES [TEL_TYPES ...]] [--allowed_tels,-a ALLOWED_TELS [ALLOWED_TELS ...]]
                [--size_cut,-z SIZE_CUT] [--leakage_cut,-l LEAKAGE_CUT] [--multiplicity_cut,-u MULTIPLICITY_CUT] [--num_epochs,-e NUM_EPOCHS] [--batch_size,-b BATCH_SIZE] [--random_seed,-s RANDOM_SEED]
                [--log_to_file] [--save2onnx] [--debug]

optional arguments:
  ``-h, --help``\
                        show this help message and exit
  ``--config_file,-c CONFIG_FILE``\
                        Path to YAML configuration file with training options
  ``--input,-i INPUT [INPUT ...]``\
                        Input directories (not required when file_list is set in the config file)
  ``--pattern,-p PATTERN [PATTERN ...]``\
                        Pattern to mask unwanted files from the data input directory
  ``--mode,-m MODE``\
                        Mode to run CTLearn; valid options: train, predict, or train_and_predict
  ``--output,-o OUTPUT``\
                        Output directory, where the logging, model weights and processed output files are stored
  ``--reco,-r RECO [RECO ...]``\
                        Reconstruction task to perform; valid options: particletype, energy, and/or direction
  ``--default_model,-d DEFAULT_MODEL``\
                        Default CTLearn Model; valid options: TRN (mono), stackedTRN (stereo), and CNNRNN (stereo)
  ``--clean, --no-clean``\
                        Flag, if the network should be trained with cleaned images (default: False)
  ``--pretrained_weights,-w PRETRAINED_WEIGHTS``\
                        Path to the pretrained weights
  ``--prediction_directory,-y PREDICTION_DIRECTORY``\
                        Path to store the CTLearn predictions (optional)
  ``--tel_types,-t TEL_TYPES [TEL_TYPES ...]``\
                        Selection of telescope types; valid option: LST_LST_LSTCam, LST_MAGIC_MAGICCam, MST_MST_FlashCam, MST_MST_NectarCam, SST_1M_DigiCam, SST_SCT_SCTCam, and/or SST_ASTRI_ASTRICam
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
  ``--save2onnx``\
                        Save model in an ONNX file
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

Build IRFs
----------

Build IRFs and sensitivity curves from CTLearn DL2-like files using pyirf:

.. code-block:: bash

   build_irf [-h] [--input INPUT [INPUT ...]] [--pattern PATTERN [PATTERN ...]] [--output OUTPUT] [--energy_range ENERGY_RANGE [ENERGY_RANGE ...]]
                 [--theta_range THETA_RANGE [THETA_RANGE ...]] [--obstime OBSTIME] [--alpha ALPHA] [--max_bg_radius MAX_BG_RADIUS] [--max_gh_cut_eff MAX_GH_CUT_EFF]
                 [--gh_cut_eff_step GH_CUT_EFF_STEP] [--init_gh_cut_eff INIT_GH_CUT_EFF] [--quality_cuts QUALITY_CUTS] [--size_cut SIZE_CUT [SIZE_CUT ...]]
                 [--leakage_cut LEAKAGE_CUT [LEAKAGE_CUT ...]]

optional arguments:
  ``-h, --help``\
                        show this help message and exit
  ``--input,-i INPUT [INPUT ...]``\
                        Input directories; default is ./
  ``--pattern,-p PATTERN [PATTERN ...]``\
                        Pattern to mask unwanted files from the data input directory; default is *.h5
  ``--output,-o OUTPUT``\
                        Output file; default is ./pyirf.fits.gz
  ``--energy_range,-e ENERGY_RANGE [ENERGY_RANGE ...]``\
                        Energy range in TeV; default is [0.03, 30.0]
  ``--theta_range,-t THETA_RANGE [THETA_RANGE ...]``\
                        Theta cut range in deg; default is [0.05, 0.3]
  ``--obstime OBSTIME``\
                        Observation time in hours; default is 50
  ``--alpha ALPHA``\
                        Scaling between on and off region; default is 0.2
  ``--max_bg_radius MAX_BG_RADIUS``\
                        Maximum background radius in deg; default is 1.0
  ``--max_gh_cut_eff MAX_GH_CUT_EFF``\
                        Maximum gamma/hadron cut efficiency; default is 0.9
  ``--gh_cut_eff_step GH_CUT_EFF_STEP``\
                        Gamma/hadron cut efficiency step; default is 0.01
  ``--init_gh_cut_eff INIT_GH_CUT_EFF``\
                        Initial gamma/hadron cut efficiency; default is 0.4
  ``--quality_cuts,-c QUALITY_CUTS``\
                        String of the quality cuts
  ``--size_cut,-z SIZE_CUT [SIZE_CUT ...]``\
                        Minimum size values
  ``--leakage_cut,-l LEAKAGE_CUT [LEAKAGE_CUT ...]``\
                        Maximum leakage2 intensity values
