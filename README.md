# CTLearn: Deep Learning for IACT Event Reconstruction

[![Build Status](https://travis-ci.com/ctlearn-project/ctlearn.svg?branch=master)](https://travis-ci.com/ctlearn-project/ctlearn)

![CTLearn Logo](images/CTLearnTextCTinBox_WhiteBkgd.png)

CTLearn is a package under active development to run deep learning models to analyze data from all major current and future arrays of Imaging Atmospheric Cherenkov Telescopes (IACTs). CTLearn v0.3.0 can load data from [CTA](https://www.cta-observatory.org/) (Cherenkov Telescope Array), [FACT](https://www.isdc.unige.ch/fact/), [H.E.S.S.](https://www.mpi-hd.mpg.de/hfm/HESS/), [MAGIC](https://magic.mpp.mpg.de/), and [VERITAS](https://veritas.sao.arizona.edu/) telescopes processed using [DL1DataHandler v0.7.3+](https://github.com/cta-observatory/dl1-data-handler).

## Install CTLearn

### Clone Repository with Git

Clone the CTLearn repository:

```bash
cd </installation/path>
git clone https://github.com/ctlearn-project/ctlearn.git
```

### Install Package with Anaconda

Next, download and install [Anaconda](https://www.anaconda.com/download/), or, for a minimal installation, [Miniconda](https://conda.io/miniconda.html). Create a new conda environment that includes all the dependencies for CTLearn:

```bash
conda env create -f </installation/path>/ctlearn/environment-<MODE>.yml
```

where `<MODE>` is either 'cpu' or 'gpu' (for linux systems) or 'macos' (for macOS systems), denoting the TensorFlow version to be installed. If installing the GPU version of TensorFlow, verify that your system fulfills all the requirements [here](https://www.tensorflow.org/install/install_linux#NVIDIARequirements). Note that there is no GPU-enabled TensorFlow version for macOS yet.

Finally, install CTLearn into the new conda environment with pip:

```bash
source activate ctlearn
cd </installation/path>/ctlearn
pip install --upgrade .
```
NOTE for developers: If you wish to fork/clone the repository and edit the code, either install with `pip -e` or reinstall  after making changes for them to take effect.

### Dependencies

- Python 3.7.3
- TensorFlow 1.13.1
- DL1DataHandler
- NumPy
- PyYAML
- Libraries used only in plotting scripts (optional)
  - Matplotlib
  - Pandas
  - Scikit-learn
  
## Download Data

CTLearn can load and process data in the HDF5 PyTables format produced from simtel files by [DL1DataHandler](https://github.com/cta-observatory/dl1-data-handler).

## Configure a Run

CTLearn encourages reproducible training and prediction by keeping all run settings in a single YAML configuration file, organized into the sections listed below. The [example config file](config/example_config.yml) describes every available setting and its possible values in detail.

### Logging

Specify model directory to store TensorFlow checkpoints and summaries, a timestamped copy of the run configuration, and optionally a timestamped file with logging output.

### Data

Describe the dataset to use and relevant settings for loading and processing it. The parameters in this section are used to initialize a DL1DataReader, which loads the data files, maps the images from vectors to arrays, applies preprocessing, and returns the data as an iterator. Data can be loaded in three modes:
- Mono: single images of one telescope type
- Stereo: events of one telescope type
- Multi-stereo: events including multiple telescope types

Parameters in this section include telescope IDs to select, auxiliary parameters to return, pre-selection cuts, image mapping settings, and pre-processing to apply to the data. Image mapping is performed by the DL1DataReader and maps the 1D pixel vectors in the raw data into 2D images. The available mapping methods are oversampling, nearest interpolation, rebinning, bilinear interpolation and bicubic interpolation, image shifting, and axial addressing.
Pre-processing is performed using the DL1DataHandler Transform class.

### Input
Set parameters of the TensorFlow Estimator input function that converts the loaded, processed data into tensors.

### Model

CTLearn works with any TensorFlow model obeying the signature `logits = model(features, params, example_description, training)` where `logits` is a vector of raw (non-normalized, pre-Softmax) predictions, `features` is a dictionary of tensors, `params` is a dictionary of model parameters, `example_description` is a DL1DataReader example description, and `training` is a Boolean that's True in training mode and False in testing mode.

To use a custom model, provide in this section the directory containing a Python file that implements the model and the module name (that is, the file name minus the .py extension) and name of the model function within the module.

In addition, CTLearn includes three [models](models) for gamma/hadron classification. CNN-RNN and Variable Input Network perform array-level classification by feeding the output of a CNN for each telescope into either a recurrent network, or a convolutional or fully-connected network head, respectively. Single Tel classifies single telescope images using a convolutional network. All three models are built on a simple, configurable convolutional network called Basic.

The values in the data to be used as labels and lists of class names where applicable are also provided in this section.

### Model Parameters

This section in its entirety is directly included as the model `params`, enabling arbitrary configuration parameters to be passed to the provided model.

### Training

Set training parameters such as the training/validation split, the number of validations to run, and how often to evaluate on the validation set, as well as hyperparameters including the base learning rate and optimizer.

### Prediction

Specify prediction settings such as the path to write the prediction file and whether to save the labels and example identifiers along with the predictions.

### TensorFlow

Set whether to run TensorFlow in debug mode.

## Run a Model

Run CTLearn from the command line:

```bash
CTLEARN_DIR=</installation/path>/ctlearn/ctlearn
python $CTLEARN_DIR/run_model.py myconfig.yml [--mode <MODE>] [--debug] [--log_to_file]
```
`--mode <MODE>`: Set run mode with `<MODE>` as `train`, `predict`, or `load_only`. If not set, defaults to `train`.

`--debug`: Set logging level to DEBUG.

`--log_to_file`: Save CTLearn logging messages to a timestamped file in the model directory instead of printing to stdout.

Alternatively, import CTLearn as a module in a Python script:

```python
import yaml
from ctlearn.run_model import run_model

with open('myconfig.yml', 'r') as myconfig:
  config = yaml.load(myconfig)
run_model(config, mode='train', debug=True, log_to_file=True)
```

View training progress in real time with TensorBoard: 

```bash
tensorboard --logdir=/path/to/my/model_dir
```

## Inspect Data

Print dataset statistics only, without running a model:

```bash
python $CTLEARN_DIR/run_model.py myconfig.yml --mode load_only
```

## Supplementary Scripts

- **plot_classifier_values.py** Plot a histogram of gamma/hadron classification values from a CTLearn predictions file.
- **plot_roc_curves.py** Plot gamma/hadron classification ROC curves from a list of CTLearn predictions files.
- **run_multiple_configurations.py** Generate a list of configuration combinations and run a model for each, for example, to conduct a hyperparameter search or to automate training or prediction for a set of models. Parses a standard CTLearn configuration file with two additional sections for Multiple Configurations added. Has an option to resume from a specific run in case the execution is interrupted.
- **auto_configuration.py** Fill the path information specific to your computer and run this script from a folder with any number of configuration files to automatically overwrite them.
- **summarize_results.py** Run this script from the folder containing the `runXX` folders generated by the `run_multiple_configurations.py` script to generate a `summary.csv` file with key validation metrics after training of each run.

## CTLearn v0.2.0 Benchmarks

Configuration files and corresponding results showing CTLearn's operation for training both single telescope and array models using simulations from all CTA telescopes are provided in [config/v0_2_0_benchmarks](config/v_0_2_0_benchmarks).

## Uninstall CTLearn

### Remove Anaconda Environment

First, remove the conda environment in which CTLearn is installed and all its dependencies:

```bash
conda remove --name ctlearn --all
```

### Remove CTLearn

Next, completely remove CTLearn from your system:

```bash
rm -rf </installation/path>/ctlearn
```
