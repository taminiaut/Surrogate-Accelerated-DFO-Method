#!/usr/bin/python3
# -------------------------- #
#  Neural network surrogate  #
# -------------------------- #

""" Imports """
import numpy as np
from ClassModel import MyModel

""" Parameters """
np.random.seed(0)  # Set the seed value for reproducibility

# NeuralNet weights are vectorized as follow :
# W1 = fullWeight[:m*n]
# b1 = fullWeight[m*n:m*(n+1)]
# W2 = fullWeight[-m-1:-1]
# b2 = fullWeight[-1]

""" Main code """
class MyNeuralNet(MyModel):
    
    # Initialize the neural network
    def __init__(self, n, f0, x0, activationType, simplexMemory, regParam, ratioHiddenInputSize, maxIterTraining):
        super().__init__(n, ratioHiddenInputSize*n, f0, x0, simplexMemory)

        self.activationType = activationType
        self.maxIterTraining = maxIterTraining
        self.regParam = regParam

        if activationType == "sigmoid":
            self.activation = lambda x: np.exp(-np.logaddexp(0, -x))
            self.activationD1 = lambda x: np.exp(-np.logaddexp(0, -x)) * (1 - np.exp(-np.logaddexp(0, -x)))
            self.activationD2 = lambda x: np.exp(-np.logaddexp(0, -x)) * (1 - np.exp(-np.logaddexp(0, -x))) * (1 - 2*np.exp(-np.logaddexp(0, -x)))

        elif activationType == "arctan":
            self.activation = lambda x: np.arctan(x)
            self.activationD1 = lambda x: 1 / (x*x+1)
            self.activationD2 = lambda x: -2*x / (x*x+1)**2

        elif activationType == "softplus":
            self.activation = lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x,0)
            self.activationD1 = lambda x: np.exp(-np.logaddexp(0, -x))
            self.activationD2 = lambda x: np.exp(-np.logaddexp(0, -x)) * (1 - np.exp(-np.logaddexp(0, -x)))

        elif activationType == "silu":
            self.activation = lambda x: x * np.exp(-np.logaddexp(0, -x))
            self.activationD1 = lambda x: np.exp(-np.logaddexp(0, -x)) + x * np.exp(-np.logaddexp(0, -x)) * (1 - np.exp(-np.logaddexp(0, -x)))
            self.activationD2 = lambda x: 2*np.exp(-np.logaddexp(0, -x)) * (1 - np.exp(-np.logaddexp(0, -x))) + x * np.exp(-np.logaddexp(0, -x)) * (1 - np.exp(-np.logaddexp(0, -x))) * (1 - 2*np.exp(-np.logaddexp(0, -x)))

        else:
            raise Exception("Unknown activation function (MyNeuralNet/init)")
        
        guessMaxW1 = np.sqrt(3/self.netDim[0])
        guessMaxW2 = np.sqrt(3/self.netDim[1])

        if activationType == "softplus" or activationType == "silu":
            guessMaxW1 *= np.sqrt(2)
            guessMaxW2 *= np.sqrt(2)
        
        self.W1 = np.random.uniform(low=-guessMaxW1,high=guessMaxW1,size=(self.netDim[1],self.netDim[0]))
        self.W2 = np.random.uniform(low=-guessMaxW2,high=guessMaxW2,size=(self.netDim[1],))

        self.b1 = np.zeros(self.netDim[1])
        self.b2 = np.zeros(1)

    # Evaluate the value of the current model
    def __call__(self, x):
        return np.einsum("k,...k->...",self.W2,self.activation(np.einsum("ij,...j->...i",self.W1,x)+self.b1))+self.b2

    # Evaluate the gradient (with respect to input) of the current model
    def grad(self, x):
        return np.einsum("lk,...l,l->...k",self.W1,self.activationD1(np.einsum("ij,...j->...i",self.W1,x)+self.b1),self.W2)

    # Evaluate the loss function
    def evalLoss(self, fullWeight):
        n, m = self.netDim[:2]
        tl = self.trainLength
        W1Matrix = np.reshape(fullWeight[:m*n],(m,n))

        hidden = np.matmul(self.trainInput[:tl[0],:],W1Matrix.T) + fullWeight[m*n:m*(n+1)]
        loss = 1/tl[0] * np.linalg.norm(np.matmul(self.activation(hidden),fullWeight[-m-1:-1])+fullWeight[-1]-self.trainOutput[:tl[0]])**2

        if tl[1] > 0:
            sobolevHidden = np.matmul(self.trainInputDerivative[:tl[1],:],W1Matrix.T) + fullWeight[m*n:m*(n+1)]
            loss += 1/tl[1] * np.linalg.norm((np.matmul(self.activationD1(sobolevHidden)*fullWeight[-m-1:-1],W1Matrix)-self.trainOutputDerivative[:tl[1],:]),ord="fro")**2

        if self.regParam > 0:
            loss += self.regParam * np.linalg.norm(fullWeight)**2

        return loss
    
    # Evaluate the gradient of the loss function (with respect to parameter)
    def evalGradLoss(self, fullWeight):
        n, m = self.netDim[:2]
        tl = self.trainLength
        W1Matrix = np.reshape(fullWeight[:m*n],(m,n))
        gradLoss = np.zeros(m*n+2*m+1)

        hidden = np.matmul(self.trainInput[:tl[0],:],W1Matrix.T) + fullWeight[m*n:m*(n+1)]
        residual1 = 2/tl[0] *(np.matmul(self.activation(hidden),fullWeight[-m-1:-1])+fullWeight[-1] - self.trainOutput[:tl[0]])
        residual2 = (self.activationD1(hidden)*fullWeight[-m-1:-1]).T * residual1

        gradLoss[:m*n] = np.reshape( np.matmul(residual2,self.trainInput[:tl[0],:]), (m*n))
        gradLoss[m*n:m*(n+1)] = np.sum(residual2,axis=1)
        
        gradLoss[-m-1:-1] = np.matmul(residual1,self.activation(hidden))
        gradLoss[-1] = np.sum(residual1)

        if tl[1] > 0:
            sobolevHidden = np.matmul(self.trainInputDerivative[:tl[1],:],W1Matrix.T) + fullWeight[m*n:m*(n+1)]
            
            sobolevResidual1 = 2/tl[1]*(np.matmul(self.activationD1(sobolevHidden)*fullWeight[-m-1:-1],W1Matrix) - self.trainOutputDerivative[:tl[1],:])
            sobolevResidual2 = (np.matmul(sobolevResidual1,(W1Matrix.T*fullWeight[-m-1:-1])) * self.activationD2(sobolevHidden)).T
            sobolevResidual3 = np.matmul(sobolevResidual1.T,self.activationD1(sobolevHidden))

            gradLoss[:m*n] += np.reshape(np.matmul(sobolevResidual2,self.trainInputDerivative[:tl[1],:]) + (sobolevResidual3*fullWeight[-m-1:-1]).T, (m*n,))
            gradLoss[m*n:m*(n+1)] += np.sum(sobolevResidual2,axis=1)
            gradLoss[-m-1:-1] += np.sum(W1Matrix.T*sobolevResidual3,axis=0)

        if self.regParam > 0:
            gradLoss += 2*self.regParam * fullWeight

        return gradLoss

    # Train the neural network on the current set of points
    def train(self, rtolTraining):
        n, m = self.netDim[:2]

        # Take as starting point the previous neural network
        fullWeight = np.zeros(m*n+2*m+1)
        fullWeight[:m*n] = np.reshape(self.W1,(m*n,))
        fullWeight[m*n:m*(n+1)] = self.b1
        fullWeight[-m-1:-1] = self.W2
        fullWeight[-1] = self.b2

        gradLossNorm, fullWeight = super().train(fullWeight,rtolTraining)

        # Update the parameters of the neural network
        self.W1 = np.reshape(fullWeight[:m*n],(m,n))
        self.b1 = fullWeight[m*n:m*(n+1)]
        self.W2 = fullWeight[-m-1:-1]
        self.b2 = fullWeight[-1]

        return gradLossNorm




        
