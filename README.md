# Deep Learning for CTA Analysis

[![Build Status](https://travis-ci.org/bryankim96/deep-learning-CTA.svg?branch=master)](https://travis-ci.org/bryankim96/deep-learning-CTA) [![Coverage Status](https://coveralls.io/repos/github/bryankim96/deep-learning-CTA/badge.svg?branch=master)](https://coveralls.io/github/bryankim96/deep-learning-CTA?branch=master) [![Code Health](https://landscape.io/github/bryankim96/deep-learning-CTA/master/landscape.svg?style=flat)](https://landscape.io/github/bryankim96/deep-learning-CTA/master)



Deep learning models for analysis and classification of image data for [CTA](https://www.cta-observatory.org/) (the Cherenkov Telescope Array).

## Installation

To install Tensorflow, follow the [instructions](https://www.tensorflow.org/install/). 
Tensorflow with GPU support must be installed to train models on GPU.

A data file of CTA data or simulations must be available in HDF5 format. 
The [ImageExtractor](https://github.com/bryankim96/image-extractor) package is available to process, calibrate, and write CTA simtel files into the HDF5 format required by the scripts here.

### Package Install

Install dependencies with:

```bash
pip install -r requirements.txt
```

Install with pip:

```bash
pip install .
```

### Package Install w/ Anaconda

Setup Anaconda environment with:

```bash
conda create -n [ENV_NAME] --file requirements.txt
source activate [ENV_NAME]
```

Install package with pip:

```bash
/path/to/anaconda/install/envs/[ENV_NAME]/bin/pip install .
```
where /path/to/anaconda/install is the path to your anaconda installation directory and ENV\_NAME is the name of your environment.

The path to the environment directory for the environment you wish to install into can be found quickly by running

```bash
conda env list
```

## Dependencies

- Python 3.6
- Tensorflow 1.4
- Pytables
- Numpy

## Configuration

All options for training a model are set by a single configuration file. 
See example_config.ini for an explanation of all available options.

**Data**
The only currently accepted data format is HDF5.
A path to a single data file containing separate groups for the training and validation sets must be provided.

**Data Processing**
Because the size of the full dataset may be very large, only a set of event indices is held in memory.
During each epoch of training, a specified number of event examples is randomly drawn from the training dataset.
Until the total number is reached, batches of a specified size are loaded and used to train the model.
Batch loading of data may be parallelized using a specified number of threads.
After every specified number of training epochs, the model is evaluated on the entire training set and validation set.

**Model**
A multiply-layered model is used to handle data from an array of telescopes.
A CNN block processes each telescope image.
For telescopes that didn't trigger, the output from the corresponding CNN is dropped out.
The output from each telescope network is then combined into a single vector or stack of feature maps.
The combined features are then processed by a network head.

Available CNN Blocks: AlexNet, MobileNet, ResNet

Available Network Heads: AlexNet, MobileNet, ResNet

**Training**
To normalize the effects of non-triggering telescopes, each batch's learning rate is scaled by the inverse of its proportion of triggering telescopes.
Model hyperparameters that can be specified are the base learning rate and the batch norm decay parameter.

**Logging**
Tensorflow checkpoints and summaries are saved to the specified directory, as is a copy of the configuration file.

## Usage

To train a model, run `python train.py myconfig.ini`. 
The model's progress can be viewed in real time using Tensorboard: `tensorboard --logdir=/path/to/my/model_dir`.

## Links

- [Cherenkov Telescope Array (CTA)](https://www.cta-observatory.org/)
- [ImageExtractor](https://github.com/bryankim96/image-extractor) 
