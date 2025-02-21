## Surrogate-accelerated DFO method ##

This repository contains the implementation of the derivative-free optimization (DFO) method proposed in [1], available on arxiv at  https://arxiv.org/abs/2502.07435. The method is based on the quadratic regularization method with finite-difference gradient proposed in [2] and we added an extrapolation procedure provided by a surrogate model of the objective function to reduce the required number of function evaluations.

The code is made of 5 files written in python :
- methodWithSurrogate.py 
- classModel.py
- classNeuralNet.py
- classRadialBasis.py
- example.py

The file "example.py" is an example of call to the surrogate-accelerated derivative-free method. "classModel.py" describe a general class of surrogate model with the handling of training data and a method for minimizing the loss function. The files "classNeuralNet.py" and "classRadialBasis.py" contain respectively the implementation of the shallow neural network and the radial basis function surrogates. The main code corresponding to the surrogate-accelerated DFO method is included in "methodWithSurrogate.py".

The argument of the surrogate-accelerated DFO method are
- f : the function to minimize
- x0 : the starting point of the method
- evalMax : the maximum number of allowed function evaluations
- tol : the tolerance of the method
- model : the type of surrogate (radial basis or neural network)
- sobolev : the indication to use or not Sobolev training


[1] T. Taminiau, E. Massart, G. N. Grapiglia, Enhancing finite-difference based derivative-free optimization methods with machine learning, 2025,  https://arxiv.org/abs/2502.07435

[2] Geovani N. Grapiglia. Worst-case evaluation complexity of a derivative-free quadratic regularization method. Optimization Letters, 18:1–19, 2024.