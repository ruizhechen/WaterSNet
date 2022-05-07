# WaterSNet

A Robust Object Segmentation Network for Under Water Scenes. Accepted by ICASSP 2022. [[paper](https://ieeexplore.ieee.org/document/9746176)]

#### Usage:

The configuration file is ```config.yaml```. You can modify this file to change Hyperparameters. 

The code is in the ```scripts``` directory.

##### Train:

run ```python train.py``` to train the model.

##### test:

run ```python test.py``` to obtain the testing results of the model as well as evaluation metrics.

Our pre-trained model: [[Pre-trained Model](https://github.com/ruizhechen/WaterSNet/releases/download/Pre-trained/pretrained_model.pth)]

#### Correction of a formula error in the paper:

Equation 5 should be ![F](http://latex.codecogs.com/svg.latex?A_2=\mathcal{F}^{-1}([M_{\beta}\otimes\mathcal{F}^{A}(A^{'}_{1})\oplus(1-M_{\beta})\otimes\mathcal{F}^{A}(A_{1}),\mathcal{F}^{P}(A_{1})]))

where ![F](http://latex.codecogs.com/svg.latex?\mathcal{F}^{A}) and ![F](http://latex.codecogs.com/svg.latex?\mathcal{F}^{P}) represent the amplitude and phase components of the Fourier transform.
