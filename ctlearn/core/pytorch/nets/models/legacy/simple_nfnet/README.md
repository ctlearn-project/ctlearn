# Simple NFNet PyTorch Implementation

This repository contains **a simple PyTorch code for Normalizer-Free Network (NFNet)**.

- Andrew Brock et al, ["Characterizing signal propagation to close the performance gap in unnormalized ResNets,"](https://arxiv.org/abs/2102.06171) ICLR 2021.
- Andrew Brock et al, ["High-Performance Large-Scale Image Recognition Without Normalization"](https://arxiv.org/abs/2102.06171), Arxiv

I implemented this code by referring [benjs's implementation code](https://github.com/benjs/nfnets_pytorch). 
This code is for training NFNet for **CIFAR-10** dataset.


## Dependency

- Python 3.7.1
- PyTorch 1.7.1
- torchvision 0.8.2


## Training

```
# Training
CUDA_VISIBLE_DEVICES=0 python main.py
```

If you want to train the model using other hyperparameters, please check argparse in ```main.py```.


## TODO

- [ ] Report Experimental results. (Accuracy for CIFAR-10)


## Acknowledgements

I referred to the following implementation codes:

- [Official codes](https://github.com/deepmind/deepmind-research/tree/master/nfnets)
- [benjs's PyTorch implementation codes](https://github.com/benjs/nfnets_pytorch)