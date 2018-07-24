# CTLearn: Deep Learning for IACT Analysis

[![Build Status](https://travis-ci.com/ctlearn-project/ctlearn.svg?branch=master)](https://travis-ci.com/ctlearn-project/ctlearn)
[![Code Health](https://landscape.io/github/ctlearn-project/ctlearn/master/landscape.svg?style=flat)](https://landscape.io/github/ctlearn-project/ctlearn/master)

CTLearn is a package for running deep learning models to perform data analysis for Imaging Atmospheric Cherenkov Telescopes. CTLearn can be used with data from [CTA](https://www.cta-observatory.org/) (the Cherenkov Telescope Array) and [VERITAS](https://veritas.sao.arizona.edu/).

## Installation

### Clone Repository with Git (recommended)

Clone the CTLearn repository:

```bash
cd </the/installation/path>
git clone https://github.com/ctlearn-project/ctlearn.git
```

### Install Package with Anaconda

Next, download and install Anaconda following the instructions [here](https://www.anaconda.com/download/). Create a new conda environment for CTLearn:

```bash
conda env create -f environment-<MODE>.yml
```

where <MODE> is either 'cpu' or 'gpu', denoting the TensorFlow version to be installed. If installing the GPU version of TensorFlow, verify that your system fulfills all the requirements [here](https://www.tensorflow.org/install/install_linux#NVIDIARequirements).

Finally, install CTLearn into the new conda environment with pip:

```bash
source activate ctlearn
cd </the/installation/path>/ctlearn
pip install --upgrade .
```
NOTE for developers: If you wish to fork/clone the respository and make changes to any of the ctlearn modules, the package must be reinstalled for the changes to take effect.

## Dependencies

- Python 3.6.5
- TensorFlow 1.9.0
- NumPy
- OpenCV
- PyTables
- PyYAML
- SciPy
- Libraries used only in plotting scripts (optional)
  - Matplotlib
  - Pillow
  - Scikit-learn

## Configuration

All options for training a model are set by a single configuration file. 
See example_config.ini for an explanation of all available options.

**Data**
The only currently accepted data format is HDF5/Pytables.
A file list containing the paths to a set of HDF5 files containing the data must be provided. The [ImageExtractor](https://github.com/cta-observatory/image-extractor) package is available to process, calibrate, and write CTA simtel files into the HDF5 format required by the scripts here. HDF5 files should be in the standard format specified by ImageExtractor.

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

## Package Removal

### Remove Anaconda Environment

Remove the conda environment in which CTLearn is installed and all its dependencies:

```bash
conda remove --name ctlearn --all
```

### Remove CTLearn

Completely remove CTLearn from your system:

```bash
rm -rf </the/installation/path>/ctlearn
```

## Links

- [Cherenkov Telescope Array (CTA)](https://www.cta-observatory.org/)
- [ImageExtractor](https://github.com/cta-observatory/image-extractor) 
