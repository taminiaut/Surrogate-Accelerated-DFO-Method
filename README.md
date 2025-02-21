## Surrogate-accelerated finite-difference gradient method ##

This repository contains the implementation of the surrogate-accelerated derivative-free optimization method proposed in "T. Taminiau, E. Massart, G. N. Grapiglia, Enhancing finite-difference based derivative-free optimization methods with machine learning", available on arxiv:  https://arxiv.org/abs/2502.07435.

The code is made of 5 files written in python :
- MethodWithSurrogate.py 
- ClassModel.py
- ClassNeuralNet.py
- ClassRadialBasis.py
- Example.py

The file "Example.py" is an example of call to the surrogate-accelerated derivative-free method. "ClassModel.py" describe a general class of surrogate model with the handling of training data and a method for minimizing the loss function. The files "ClassNeuralNet.py" and "ClassRadialBasis.py" contain respectively the implementation of the shallow neural network and radial basis function surrogates. The main code corresponding to the surrogate-accelerated method is included in "MethodWithSurrogate.py".