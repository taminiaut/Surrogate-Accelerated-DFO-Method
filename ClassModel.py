#!/usr/bin/python3
# ------------------------------------------- #
#  Abstract model for surrogate optimization  #
# ------------------------------------------- #

""" Imports """
import numpy as np
from time import perf_counter

""" Parameters """
separableTol = 1e-16  # Minimal distance between training data
maxIterLineSearch = 500  # Wolfe linesearch parameter 
c1 = 1e-4  # Wolfe linesearch parameter 
c2 = 0.5  # Wolfe linesearch parameter 

""" Main code """
class MyModel():

    # Initialize the surrogate model
    def __init__(self, n, m, f0, x0, simplexMemory):
        self.netDim = np.array([n, m, 1], dtype=int)
        self.netMemory = (simplexMemory*(n+1), simplexMemory)
        
        # Store training function values
        self.trainInput = np.zeros((self.netMemory[0],n))
        self.trainOutput = np.zeros(self.netMemory[0])
        
        self.trainInput[0,:] = x0
        self.trainOutput[0] = f0

        # Store training derivative values
        self.trainInputDerivative = np.zeros((self.netMemory[1],n))
        self.trainOutputDerivative = np.zeros((self.netMemory[1],n))
    
        # Remember the current ammount of data and the last index added
        self.trainLength = np.array([1, 0],dtype=int)
        self.trainLast = np.array([1, 0],dtype=int)

    # Add new training data either function or derivative values
    def addTrainingData(self, sampleValue, samplePoint, isDerivative=False):
        
        # Check if the new points are not too close with respect to the current data
        if not isDerivative:
            check = np.min(np.linalg.norm(samplePoint[:,np.newaxis,:]-self.trainInput[np.newaxis,:self.trainLength[0],:],axis=-1),axis=1) >= separableTol
            sampleValue = sampleValue[check]
            samplePoint = samplePoint[check,:]
        elif isDerivative and self.trainLength[1] > 0:
            check = np.min(np.linalg.norm(samplePoint[:,np.newaxis,:]-self.trainInputDerivative[np.newaxis,:self.trainLength[1],:],axis=-1),axis=1) >= separableTol
            sampleValue = sampleValue[check,:]
            samplePoint = samplePoint[check,:]
        
        # Check if the dimensions match
        pointShape = np.shape(samplePoint)
        valueShape = np.shape(sampleValue)
        if not isDerivative:
            if pointShape[0] != valueShape[0] or pointShape[1] != self.netDim[0]:
                raise Exception("Training data dimensions are not correct (MyModel/addTrainingData)")
            size = valueShape[0]
        else:

            if len(pointShape) == 1:
                if pointShape[0] != valueShape[0] or pointShape[0] != self.netDim[0]:
                    raise Exception("Training data dimensions are not correct (MyModel/addTrainingData)")
                size = 1
            else:
                if pointShape[0] != valueShape[0] or pointShape[1] != valueShape[1] or pointShape[1] != self.netDim[0]:
                    raise Exception("Training data dimensions are not correct (MyModel/addTrainingData)")
                size = pointShape[0]
        

        j = int(isDerivative)
        if self.trainLast[j]+size <= self.netMemory[j]:
            # New sample set does not reach the end of the queue
            if not isDerivative:
                self.trainOutput[self.trainLast[j]:self.trainLast[j]+size] = sampleValue
                self.trainInput[self.trainLast[j]:self.trainLast[j]+size,:] = samplePoint

            else:
                self.trainOutputDerivative[self.trainLast[j]:self.trainLast[j]+size,:] = sampleValue
                self.trainInputDerivative[self.trainLast[j]:self.trainLast[j]+size,:] = samplePoint

        else:
            # New sample set overcomes the end of the queue
            if size > self.netMemory[j]:
                raise Exception("Not enough memory to store all the sample points (MyModel/addTrainingData)")

            rest = self.trainLast[j]+size - self.netMemory[j]
            if not isDerivative:
                self.trainOutput[self.trainLast[j]:] = sampleValue[:size-rest]
                self.trainOutput[:rest] = sampleValue[size-rest:]

                self.trainInput[self.trainLast[j]:,:] = samplePoint[:size-rest]
                self.trainInput[:rest,:] = samplePoint[size-rest:]

            else:
                self.trainOutputDerivative[self.trainLast[j]:,:] = sampleValue[:size-rest]
                self.trainOutputDerivative[:rest,:] = sampleValue[size-rest:]

                self.trainInputDerivative[self.trainLast[j]:,:] = samplePoint[:size-rest]
                self.trainInputDerivative[:rest,:] = samplePoint[size-rest:]

        # Update current last point added and training length
        self.trainLast[j] = (self.trainLast[j]+size) % self.netMemory[j]
        self.trainLength[j] = min(self.trainLength[j]+size,self.netMemory[j])
        
    # Train the neural network on the current set of points
    def train(self, fullWeight, rtolTraining):
        tStart = perf_counter()
        nParameter = len(fullWeight)

        loss = self.evalLoss(fullWeight)
        gradLoss = self.evalGradLoss(fullWeight)
        searchDirection = -gradLoss
        
        gradLossNorm = np.linalg.norm(gradLoss)
        atolTraining = max(gradLossNorm, 1) * rtolTraining

        M = 10
        S = np.zeros((M,nParameter))
        Y = np.zeros((M,nParameter))
        rho = np.zeros(M)
        a = np.zeros(M)

        for k in range(self.maxIterTraining):
            
            # Strong wolfe line search
            alpha = 1
            alphaInterval = np.array([0, np.inf])
            armijoLoss = loss

            directionalDerivative = np.dot(searchDirection,gradLoss)
            for l in range(maxIterLineSearch):

                newLoss = self.evalLoss(fullWeight+alpha*searchDirection)
                if newLoss > loss + c1 * alpha * directionalDerivative or newLoss > armijoLoss:
                    alphaInterval[1] = alpha
                    alpha = (alphaInterval[0] + alphaInterval[1]) / 2
                    continue

                newGradLoss = self.evalGradLoss(fullWeight+alpha*searchDirection)
                newDirectionalDerivative = np.dot(searchDirection,newGradLoss)
                if abs(newDirectionalDerivative) > - c2 * directionalDerivative and abs(newDirectionalDerivative) > 1e-8:
                    if (alphaInterval[1] - alphaInterval[0]) * newDirectionalDerivative >= 0:
                        alphaInterval[1] = alphaInterval[0]
                    alphaInterval[0] = alpha
                    armijoLoss = newLoss

                    # Particular case when upper bound is not finite
                    if alphaInterval[1] == np.inf:
                        alpha = 2 * alphaInterval[0]
                    else:
                        alpha = (alphaInterval[0] + alphaInterval[1]) / 2
                    continue

                fullWeight = fullWeight + alpha*searchDirection
                break

            # Stop the method earlier if the training takes too much time
            if perf_counter()-tStart > 20:
                break


            if l == maxIterLineSearch-1:

                newGradLoss = self.evalGradLoss(fullWeight+alpha*searchDirection)
                
                fullWeight = fullWeight + alpha*searchDirection
                S[k%M,:] = alpha*searchDirection
                Y[k%M,:] = newGradLoss - gradLoss

                loss = self.evalLoss(fullWeight)
                gradLoss = self.evalGradLoss(fullWeight)
                gradLossNorm = np.linalg.norm(gradLoss)
                searchDirection = -gradLoss

                continue

            # Update current information
            S[k%M,:] = alpha * searchDirection
            Y[k%M,:] = newGradLoss - gradLoss

            loss = newLoss
            gradLoss = newGradLoss
            gradLossNorm = np.linalg.norm(gradLoss)
            
            # Check if new point is almost stationnary
            if gradLossNorm <= atolTraining:
                break

            # Compute LM-BFGS search direction
            M2 = min(k+1,M)
            q = np.copy(newGradLoss)
            for i in range(M2):
                rho[i] = 1 / max(np.dot(S[(k-i)%M,:],Y[(k-i)%M,:]),1e-20)
                a[i] = rho[i] * np.dot(S[(k-i)%M,:],q)
                q = q - a[i] * Y[(k-i)%M,:]
            z = q / (rho[0] * np.dot(Y[k%M,:],Y[k%M,:]))
            for i in range(M2-1,-1,-1):
                z = z + (a[i] - rho[i] * np.dot(Y[(k-i)%M,:],z)) * S[(k-i)%M,:]
            searchDirection = -z

        return gradLossNorm, fullWeight
    
