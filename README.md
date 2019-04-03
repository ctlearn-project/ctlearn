# CTLearn: Deep Learning for IACT Event Reconstruction

[![Build Status](https://travis-ci.com/ctlearn-project/ctlearn.svg?branch=master)](https://travis-ci.com/ctlearn-project/ctlearn)

![Validation Accuracy](images/CTLearnTextCTinBox_WhiteBkgd.png)

CTLearn is a package under active development to run deep learning models to analyze data from all major current and future arrays of Imaging Atmospheric Cherenkov Telescopes (IACTs). CTLearn v0.3.0 can load data from [CTA](https://www.cta-observatory.org/) (Cherenkov Telescope Array), [FACT](https://www.isdc.unige.ch/fact/), [H.E.S.S.](https://www.mpi-hd.mpg.de/hfm/HESS/), [MAGIC](https://magic.mpp.mpg.de/), and [VERITAS](https://veritas.sao.arizona.edu/) telescopes processed using [DL1DataHandler v0.6.0](https://github.com/cta-observatory/dl1-data-handler).

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
NOTE for developers: If you wish to fork/clone the respository and make changes to any of the ctlearn modules, the package must be reinstalled for the changes to take effect.

### Dependencies

- Python 3.6.5
- TensorFlow 1.12.0
- NumPy
- AstroPy
- OpenCV
- PyTables
- PyYAML
- SciPy
- Libraries used only in plotting scripts (optional)
  - Matplotlib
  - Pillow
  - Scikit-learn
  
## Download Data

CTLearn can load and process data in the HDF5 PyTables format produced from simtel files by [DL1DataHandler](https://github.com/cta-observatory/dl1-data-handler). Instructions for how to download CTA Prod3b data processed into this format are available on the [CTA internal wiki](https://forge.in2p3.fr/projects/cta_analysis-and-simulations/wiki/Machine_Learning_for_Event_Reconstruction#Common-datasets).

## Configure a Run

CTLearn encourages reproducible training and prediction by keeping all run settings in a single YAML configuration file, organized into the sections listed below. The [example config file](config/example_config.yml) describes every available setting and its possible values in detail.

### Logging

Specify model directory to store TensorFlow checkpoints and summaries, a timestamped copy of the run configuration, and optionally a timestamped file with logging output.

### Data

Describe the data to use, including the format, list of file paths, and whether to apply preprocessing. Includes subsections for **Loading** for parameters for selecting data such as the telescope type and pre-selection cuts to apply, **Processing** for data preprocessing settings such as cropping or normalization, and **Input** for parameters of the TensorFlow Estimator input function that converts the loaded, processed data into tensors. 

Data may be loaded in two ways, either event-wise in `array` mode which yields data from all telescopes in a specified array as well as auxiliary information including each telescope's position, or one image at a time in `single_tel` mode. In `array` mode, data from either a single telescope type or multiple telescope types may be loaded. 

By default, each input image has a single channel indicating integrated pulse intensity per pixel.
If the option `use_peak_times` is set to `True`, an additional channel with peak pulse arrival times per pixel will be loaded.

### Image Mapping

Set parameters for mapping the 1D pixel vectors in the raw data into 2D images, including the hexagonal grid conversion algorithm to use and how much padding to apply. The available hexagonal conversion algorithms are oversampling, nearest interpolation, rebinning, bilinear interpolation and bicubic interpolation, image shifting, and axial addressing.

### Model

CTLearn works with any TensorFlow model obeying the signature `logits = model(features, params, training)` where `logits` is a vector of raw (non-normalized, pre-Softmax) predictions, `features` is a dictionary of tensors, `params` is a dictionary of training parameters and dataset metadata, and `training` is a Boolean that's True in training mode and False in testing mode. Since models in CTLearn v0.2.0 return only a single logits vector, they can perform only one classification task (e.g. gamma/hadron classification).

Provide in this section the directory containing a Python file that implements the model and the module name (that is, the file name minus the .py extension) and name of the model function within the module. Everything in the **Model Parameters** section is directly included in the model `params`, so arbitrary configuration parameters may be passed to the provided model.

In addition, CTLearn includes three [models](models) for gamma/hadron classification. CNN-RNN and Variable Input Network perform array-level classification by feeding the output of a CNN for each telescope into either a recurrent network, or a convolutional or fully-connected network head, respectively. Single Tel classifies single telescope images using a convolutional network. All three models are built on a simple, configurable convolutional network called Basic.

### Training

Set training parameters such as the number of validations to run and how often to evaluate on the validation set, as well as, in the **Hyperparameters** section, hyperparameters including the base learning rate and optimizer.

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

## Classes

**DataLoader and HDF5DataLoader** Load a set of IACT data and provide a generator yielding NumPy arrays of examples (data and labels) as well as additional information about the dataset. HDF5DataLoader is the specifc implementation of the abstract base class DataLoader for the DL1DataHandler v0.6.0 HDF5 format. Because it's prohibitive to store a large dataset in memory, HDF5DataLoader instead provides a method `get_example_generators()` that returns functions returning generators that yield example identifiers (run number, event number, and, in `single_tel` mode, tel id) as well as the class weights, and methods `get_example()` and `get_image()` to map these identifiers to examples of data and labels and to telescope images. HDF5DataLoader also provides methods `get_metadata()` and `get_auxiliary_data()` that return dictionaries of additional information about the dataset. A DataProcessor provided either at initialization or using the method `add_data_processor()` applies preprocessing to the examples and an ImageMapper provided at initialization maps the images.

**DataProcessor** Preprocess IACT data. DataProcessor has a method `process_example()` that accepts an example of a list of NumPy arrays of data and an integer label along with the telescope type and returns preprocessed data in the same format, and a method `get_metadata()` that returns a dictionary of information about the processed data. A DataProcessor with no options set leaves the example unchanged. Preprocessing methods implemented in CTLearn v0.2.0 include cropping an image about the shower centroid and applying logarithmic normalization. 

**ImageMapper** Map vectors of pixel values (as stored in the raw data) to square camera images. This is done with the `map_image()` method that accepts a vector of pixel values and telescope type and returns the camera image converted to a square array. This is not a unique transformation for cameras with pixels laid out in a hexagonal grid, so the hexagonal conversion method is configurable. The implemented method are oversampling, nearest interpolation, rebinning, bilinear interpolation and bicubic interpolation. ImageMapper can convert data from all CTA telescope and camera combinations currently under development, as well as data from all IACTs (VERITAS, MAGIC, FACT, HESS-I and HESS-II.)

These classes may be used independently of the TensorFlow-based portion of CTLearn, e.g.:

```python
from ctlearn.data_loading import HDF5DataLoader

myfiles = ['myfile1.h5', 'myfile2.h5',...]
data_loader = HDF5DataLoader(myfiles)
train_generator, validation_generator, class_weights = data_loader.get_example_generators()
# Print a list of NumPy arrays of telescope data, a NumPy array of telescope position
# coordinates, and a binary label for the first example in the training set
example_identifiers = list(train_generator())[0]
print(data_loader.get_example(*example_identifiers))
```

## Supplementary Scripts

- **plot_classifier_values.py** Plot a histogram of gamma/hadron classification values from a CTLearn predictions file.
- **plot_roc_curves.py** Plot gamma/hadron classification ROC curves from a list of CTLearn predictions files.
- **plot_camera_image.py** Plot all cameras for all hexagonal conversion method with dummy data.
- **print_dataset_metadata.py** Print metadata for a list of ImageExtractor HDF5 files using HDF5DataLoader.
- **run_multiple_configurations.py** Generate a list of configuration combinations and run a model for each, for example, to conduct a hyperparameter search or to automate training or prediction for a set of models. Parses a standard CTLearn configuration file with two additional sections for Multiple Configurations added. Has an option to resume from a specific run in case the execution is interrupted.
- **visualize_bounding_boxes.py** Plot IACT images with overlaid bounding boxes using DataProcessor's crop method. Useful for manually tuning cropping and cleaning parameters.
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
