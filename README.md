# CTLearn: Deep Learning for IACT Analysis

[![Build Status](https://travis-ci.com/ctlearn-project/ctlearn.svg?branch=master)](https://travis-ci.com/ctlearn-project/ctlearn)

CTLearn is a package for running deep learning models to perform data analysis for Imaging Atmospheric Cherenkov Telescopes. CTLearn can load data from the [CTA](https://www.cta-observatory.org/) (Cherenkov Telescope Array) and [VERITAS](https://veritas.sao.arizona.edu/) telescopes processed using [ImageExtractor](https://github.com/cta-observatory/image-extractor).

## Install CTLearn

### Clone Repository with Git

Clone the CTLearn repository:

```bash
cd </installation/path>
git clone https://github.com/ctlearn-project/ctlearn.git
```

### Install Package with Anaconda

Next, download and install Anaconda following the instructions [here](https://www.anaconda.com/download/). Create a new conda environment for CTLearn:

```bash
conda env create -f </installation/path>/ctlearn/environment-<MODE>.yml
```

where `<MODE>` is either 'cpu' or 'gpu', denoting the TensorFlow version to be installed. If installing the GPU version of TensorFlow, verify that your system fulfills all the requirements [here](https://www.tensorflow.org/install/install_linux#NVIDIARequirements).

Finally, install CTLearn into the new conda environment with pip:

```bash
source activate ctlearn
cd </installation/path>/ctlearn
pip install --upgrade .
```
NOTE for developers: If you wish to fork/clone the respository and make changes to any of the ctlearn modules, the package must be reinstalled for the changes to take effect.

### Dependencies

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
  
## Download Data

CTLearn can load and process data in the HDF5 PyTables format produced from simtel files by [ImageExtractor](https://github.com/cta-observatory/image-extractor). Instructions for how to download CTA Prod3b data processed into this format are available on the [CTA internal wiki](https://forge.in2p3.fr/projects/cta_analysis-and-simulations/wiki/Machine_Learning_for_Event_Reconstruction).

## Configure a Run

CTLearn encourages reproducible training and prediction by keeping all run settings in a single YAML configuration file, organized into the sections listed below. The [example config file](config/example_config.yml) describes every available setting and its possible values in detail.

### Logging

Specify model directory to store TensorFlow checkpoints and summaries, a timestamped copy of the run configuration, and optionally a timestamped file with logging output.

### Data

Describe the data to use, including the format, list of file paths, and whether to apply preprocessing. Includes subsections for **Loading** for parameters for selecting data such as the telescope type and pre-selection cuts to apply, **Processing** for data preprocessing settings such as cropping or normalization, and **Input** for parameters of the TensorFlow Estimator input function that converts the loaded, processed data into tensors. 

As of CTLearn v0.2.0, only data of a single telescope type may be loaded at a time, even if the underlying dataset includes telescopes of multiple types. Data may be loaded in two ways, either event-wise in `array` mode yielding data from all telescopes in a specified array as well as auxiliary information including each telescope's position, or one image at a time in `single_tel` mode. 

### Image Mapping

Set parameters for mapping the 1D pixel vectors in the raw data into 2D images, including the hexagonal grid conversion algorithm to use and how much padding to apply. As of CTLearn v0.2.0, the only implemented hexagaonal conversion algorithm is oversampling.

### Model

CTLearn works with any TensorFlow model obeying the signature `logits = model(features, params, training)` where `logits` is a vector of raw (non-normalized, pre-Softmax) predictions, `features` is a dictionary of tensors, `params` is a dictionary of training parameters and dataset metadata, and `training` is a Boolean that's True in training mode and False in testing mode. Since models in CTLearn v0.2.0 return only a single logits vector, they can perform only one classification task (e.g. gamma/hadron classification).

Provide in this section the directory containing a Python file that implements the model and the module name (that is, the file name minus the .py extension) and name of the model function within the module. Everything in the **Model Parameters** section is directly included in the model `params`, so arbitrary configuration parameters may be passed to the provided model.

In addition, CTLearn includes three [models](models) for gamma/hadron classification. CNN-RNN and Variable Input Network perform array-level classification by feeding the output of a CNN for each telescope into either a recurrent network, or a convolutional or full-connected network head, respectively. Single Tel classifies single telescope images using a convolutional network. All three models are built on a simple, configurable convolutional network called Basic.

### Training

Set training parameters such as the number of epochs to run and how often to evaluate on the validation set, as well as, in the **Hyperparameters** section, hyperparameters including the base learning rate and optimizer.

### Prediction

Specify prediction settings such as the path to write the prediction file.

### TensorFlow

Set whether to run TensorFlow in debug mode.

## Run a Model

Run CTLearn from the command line:

```bash
CTLEARN_DIR=</installation/path>/ctlearn/ctlearn
python $CTLEARN_DIR/run_model.py myconfig.yml [--mode <MODE>] [--debug] [--log_to_file]
```
`--mode <MODE>`: Set run mode with `<MODE>` either `train` or `predict`. If not set, defaults to `train`.

`--debug`: Set logging level to DEBUG.

`--log_to_file`: Save CTLearn logging messages to a file in the model directory instead of printing to stdout.

Alternatively, import CTLearn as a module in a Python script:

```python
import yaml
from ctlearn.run_model import run_model

config = yaml.load('myconfig.yml')
run_model(config, mode='train', debug=True, log_to_file=True)
```

View training progress in real time with TensorBoard: 

```bash
tensorboard --logdir=/path/to/my/model_dir
```

## Classes

**DataLoader and HDF5DataLoader** Load a dataset.

**DataProcessor**
Because the size of the full dataset may be very large, only a set of event indices is held in memory.
During each epoch of training, a specified number of event examples is randomly drawn from the training dataset.
Until the total number is reached, batches of a specified size are loaded and used to train the model.
Batch loading of data may be parallelized using a specified number of threads.
After each training epoch, the model is evaluated on the validation set.

**ImageMapper**

## Supplementary Scripts

- **plot_classifier_values.py** Plot a histogram of gamma/hadron classification values from a CTLearn predictions file.
- **plot_roc_curves.py** Plot gamma/hadron classification ROC curves from a list of CTLearn predictions files.
- **print_dataset_metadata.py** Print metadata for a list of ImageExtractor HDF5 files using HDF5DataLoader.
- **run_multiple_configurations.py** Generate a list of configuration combinations and run a model for each, for example, to conduct a hyperparameter search or to automate training or prediction for a set of models. Parses a standard CTLearn configuration file with two additional sections for Multiple Configurations added.
- **visualize_bounding_boxes.py** Plot IACT images with overlaid bounding boxes using DataProcessor's crop method. Useful for manually tuning cropping and cleaning parameters.

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
