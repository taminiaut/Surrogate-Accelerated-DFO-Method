#!/usr/bin/python3
# --------------------------------------------------------- #
#  Surrogate-accelerated finite-difference gradient method  #
# --------------------------------------------------------- #

""" Imports """
import numpy as np
from classNeuralNet import MyNeuralNet
from classRadialBasis import MyRadialBasis

""" Parameters """
sigmaMin = 1e-2
iterLineSearchMax = 100

simplexMemory = 10  # Maximum size of the training set in simplex gradients
extrapolMax = 500  # Maximum number of extrapolation by iteration
rho = 1e-4  # Armijo linesearch parameter in extrapolation steps

reg = 1e-4  # Regularization parameter for neural network training
ratioHiddenInputSize = 5  # Ratio between input and hidden width of the neural network
rtolTraining = 1e-6  # Tolerance for training the neural network
maxIterTraining = 1000  # Maximum number of iterations for training the neural network

""" Main code """
def finiteDiffGradientWithSurrogate(f, x0, evalMax, tol, model, sobolev=True):
    n = len(x0)
    I = np.eye(n)
    sigma = 1

    f0 = f(x0)
    nEval = 1

    if model == "NeuralNet":
        # Default activation function is softplus
        model = MyNeuralNet(n,f0,x0,"softplus",simplexMemory,reg,ratioHiddenInputSize,maxIterTraining)
    elif model == "RadialBasis":
        # Default radial basis function is gaussian
        model = MyRadialBasis(n,f0,x0,"gaussian",simplexMemory)
    else:
        raise Exception("Unknown surrogate model (finiteDiffGradientWithSurrogate)")

    fSample = np.zeros(n)
    fExtrapol = np.zeros(extrapolMax)
    extrapol = np.zeros((extrapolMax,n))

    k = 0
    while True:  # Iterate until max budget of function evaluations is reached
        
        h = 2*tol / (5*sigma*np.sqrt(n))
        
        for l in range(iterLineSearchMax):

            # Evaluation at sample points
            for j in range(n):
                fSample[j] = f(x0 + h*I[:,j])
            nEval += n
            g0 = (fSample - f0) / h
            model.addTrainingData(fSample,x0+h*I,False)

            if nEval >= evalMax:
                return x0, f0
            
            if np.linalg.norm(g0) >= 4/5 * tol:

                xNew = x0 - 1/sigma * g0
                fNew = f(xNew)
                nEval += 1

                if nEval >=  evalMax:
                    return x0, f0

                if f0 - fNew >= 1/(8*sigma) * np.linalg.norm(g0)**2:
                    
                    model.addTrainingData(np.array([fNew]),np.array([xNew]),False)
                    if sobolev:
                        model.addTrainingData(np.array([g0]),np.array([x0]),True)
                    x0 = xNew
                    f0 = fNew
                    break
                
            h /= 2
            sigma *= 2

        # Train the model based on previously computed oracles
        model.train(rtolTraining)

        # Try to avoid new computations of finite-difference gradient with the model
        L = sigma
        for j in range(extrapolMax):

            # Armijo Line Search on the model
            m0 = model(x0)
            d0 = model.grad(x0)
            norm2d0 = np.linalg.norm(d0)**2
            for l in range(iterLineSearchMax):
                mNew = model(x0 - 1/L*d0)
                if mNew <= m0 - rho/L * norm2d0:
                    break
                L *= 2
            
            extrapol[j,:] = x0 - 1/L*d0
            fExtrapol[j] = f(extrapol[j,:])
            nEval += 1
            L /= 2

            if nEval >= evalMax:
                return x0, f0

            if f0 - fExtrapol[j] > 2/(25*sigma) * tol**2:
                x0 = extrapol[j,:]
                f0 = fExtrapol[j]
            else:
                if j+1 <= (n+1)*simplexMemory:
                    model.addTrainingData(fExtrapol[:j+1],extrapol[:j+1,:],False)
                # Not enough storage for all the extrapolated function values
                else:
                    model.addTrainingData(fExtrapol[-(n+1)*simplexMemory:],extrapol[-(n+1)*simplexMemory:,:],False)
                break

        sigma = max(sigma/2, sigmaMin)
        k += 1



