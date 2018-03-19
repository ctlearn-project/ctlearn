# Deep Learning for CTA Analysis

[![Build Status](https://travis-ci.org/bryankim96/ctalearn.svg?branch=master)](https://travis-ci.org/bryankim96/ctalearn) [![Coverage Status](https://coveralls.io/repos/github/bryankim96/ctalearn/badge.svg?branch=master)](https://coveralls.io/github/bryankim96/ctalearn?branch=master) [![Code Health](https://landscape.io/github/bryankim96/ctalearn/master/landscape.svg?style=flat)](https://landscape.io/github/bryankim96/ctalearn/master)




Deep learning models for analysis and classification of image data for [CTA](https://www.cta-observatory.org/) (the Cherenkov Telescope Array).

## Installation

### Package Install w/ Pip

Install other dependencies (besides Tensorflow) with:

```bash
pip install -r requirements.txt
```

Install with pip:

```bash
pip install .
```

Finally, install the CPU or GPU version of Tensorflow using the instructions [here](https://www.tensorflow.org/install/install_linux#installing_with_native_pip). 
Tensorflow with GPU support must be installed to train models on GPU.

NOTE: The current version of ctalearn uses Tensorflow 1.4.1, so use the following links to download (for Python 3.6):

CPU: https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp36-cp36m-linux_x86_64.whl  
GPU: https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp36-cp36m-linux_x86_64.whl

### Package Install w/ Anaconda (Recommended)

Setup Anaconda environment with:

```bash
conda config --add channels conda-forge
conda create -n [ENV_NAME] --file requirements.txt python=3.6
source activate [ENV_NAME]
```

Install package into the conda environment with pip:

```bash
/path/to/anaconda/install/envs/[ENV_NAME]/bin/pip install .
```
where /path/to/anaconda/install is the path to your anaconda installation directory and ENV\_NAME is the name of your environment.

The path to the environment directory for the environment you wish to install into can be found quickly by running

```bash
conda env list
```
Finally, install the CPU or GPU version of Tensorflow using the instructions [here](https://www.tensorflow.org/install/install_linux#installing_with_native_pip). 
Tensorflow with GPU support must be installed to train models on GPU.

NOTE: The current version of ctalearn uses Tensorflow 1.4.1, so use the following links to download (for Python 3.6):

CPU: https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp36-cp36m-linux_x86_64.whl  
GPU: https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.1-cp36-cp36m-linux_x86_64.whl

NOTE for developers: If you wish to fork/clone the respository and make changes to any of the ctalearn modules, the package should be reinstalled for the changes to take effect.

## Dependencies

- Python 3.6
- Tensorflow 1.4.1
- Pytables 3.4.2
- Numpy 1.14.2
- OpenCV 3.3.1

and others specified in requirements.txt

## Configuration

All options for training a model are set by a single configuration file. 
See example_config.ini for an explanation of all available options.

**Data**
The only currently accepted data format is HDF5/Pytables.
A file list containing the paths to a set of HDF5 files containing the data must be provided. The [ImageExtractor](https://github.com/bryankim96/image-extractor) package is available to process, calibrate, and write CTA simtel files into the HDF5 format required by the scripts here. HDF5 files should be in the standard format specified by ImageExtractor.

For instructions on how to download the full pre-processed Prod3b dataset in ImageExtractor HDF5 format, see the wiki page [here](https://forge.in2p3.fr/projects/cta_analysis-and-simulations/wiki/Machine_Learning_for_Event_Reconstruction). (NOTE: requires a CTA account). 

**Data Processing**
Because the size of the full dataset may be very large, only a set of event indices is held in memory.
During each epoch of training, a specified number of event examples is randomly drawn from the training dataset.
Until the total number is reached, batches of a specified size are loaded and used to train the model.
Batch loading of data may be parallelized using a specified number of threads.
After each training epoch, the model is evaluated on the validation set.

**Model**
Several higher-level model types are provided to train networks for single-telescope classification (single_tel_model) and array (multiple image) classification (variable_input_model, cnn_rnn_model)

Available CNN Blocks: Basic, AlexNet, MobileNet, ResNet, DenseNet

Available Network Heads: AlexNet (fully connected telescope combination), AlexNet (convolutional telescope combination), MobileNet, ResNet, Basic (fully connected telescope combination), Basic (convolutional telescope combination)

**Training**
Training hyperparameters including the learning rate and optimizer can be set in the configuration file.

**Logging**
Tensorflow checkpoints and summaries are saved to the specified model directory, as is a copy of the configuration file.

## Usage

To train a model, run `python train.py myconfig.ini`. 
The following flags may be set: `--debug` to set DEBUG logging level, `--log_to_file` to save logger messages to a file in the model directory.
The model's progress can be viewed in real time using Tensorboard: `tensorboard --logdir=/path/to/my/model_dir`.

## Links

- [Cherenkov Telescope Array (CTA)](https://www.cta-observatory.org/)
- [ImageExtractor](https://github.com/bryankim96/image-extractor) 
