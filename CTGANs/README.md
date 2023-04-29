# CTGANs

## Set up

First, [CTLearn](https://github.com/ctlearn-project/ctlearn) installation is required. Installation can be carried out from source:

```
CTLEARN_VER=0.6.0
git clone https://github.com/ctlearn-project/ctlearn
cd ctlearn
conda env create -f environment-gpu.yml
conda activate ctlearn
pip install ctlearn==$CTLEARN_VER
```

Or simply:

```
CTLEARN_VER=0.6.0
mode=gpu
wget https://raw.githubusercontent.com/ctlearn-project/ctlearn/v$CTLEARN_VER/environment-$mode.yml
conda env create -n [ENVIRONMENT_NAME] -f environment-$mode.yml
conda activate [ENVIRONMENT_NAME]
pip install ctlearn==$CTLEARN_VER
ctlearn -h
```

Additionally, the following installation is necessary:

1. `pip install tensorflow-addons`
2. `pip install --upgrade matplotlib`
3. `pip install wandb` (afterwards run `wandb login` and log in)

## Usage

First, update `GANs.yml` (and `predictor.yml` if no predefined model is used as a predictor). Possible labels are 'particletype', 'energy' and 'direction'. To train the models, simply run `main.py` and introduce the path to `GANs.yml`.