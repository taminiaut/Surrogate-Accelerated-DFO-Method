#!/usr/bin/python3
# ---------------------------------------------------- #
#  Example of use of the surrogate-accelerated method  #
# ---------------------------------------------------- #


import numpy as np
from methodWithSurrogate import finiteDiffGradientWithSurrogate


# Rosenbrock function
f = lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
x0 = np.array([0, 0])


tol = 1e-4
evalMax = 500
sobolev = True

x1, f1 = finiteDiffGradientWithSurrogate(f,x0,evalMax,tol,"NeuralNet",sobolev)
x2, f2 = finiteDiffGradientWithSurrogate(f,x0,evalMax,tol,"RadialBasis",sobolev)

print("Last objective value for the NN-accelerated DFO method is " + str(f1))
print("Last objective value for the RBF-accelerated DFO method is " + str(f2))

