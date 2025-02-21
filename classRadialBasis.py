#!/usr/bin/python3
# ------------------------ #
#  Radial basis surrogate  #
# ------------------------ #

""" Imports """
import numpy as np
from classModel import MyModel

""" Main code """
class MyRadialBasis(MyModel):

    # Initialize the neural network
    def __init__(self, n, f0, x0, activationType, simplexMemory):
        super().__init__(n, -1, f0, x0, simplexMemory)

        if activationType == "multiquadratic" or activationType == "invmultiquadratic":
            
            if activationType == "multiquadratic":
                sign = -1
                d = 1/2
            elif activationType == "invmultiquadratic":
                sign = 1
                d = -1/2

            self.phi     = lambda r: sign * (1+r*r)**d
            self.dxphi   = lambda r,v: sign * 2*d*v*((1+r*r)**(d-1))[:,:,np.newaxis]
            self.dyphi   = lambda r,v: sign * (-2*d*v*((1+r*r)**(d-1))[:,:,np.newaxis])
            self.dxdyphi = lambda r,v: sign * (-2*d*((1+r*r)**(d-1))[:,:,np.newaxis,np.newaxis]*np.eye(self.netDim[0])[np.newaxis,np.newaxis,:,:] \
                                     -4*d*(d-1)*((1+r*r)**(d-2))[:,:,np.newaxis,np.newaxis]*v[:,:,np.newaxis,:]*v[:,:,:,np.newaxis])
        
        elif activationType == "cubic":

            self.phi     = lambda r: r**3
            self.dxphi   = lambda r,v: 3*v*r[:,:,np.newaxis]
            self.dyphi   = lambda r,v: -3*v*r[:,:,np.newaxis]
            self.dxdyphi = lambda r,v: -3*r[:,:,np.newaxis,np.newaxis]*np.eye(self.netDim[0])[np.newaxis,np.newaxis,:,:] \
                                     -3*v[:,:,np.newaxis,:]*v[:,:,:,np.newaxis]/np.where(r>1e-30,r,1)[:,:,np.newaxis,np.newaxis]

        elif activationType == "gaussian":
            
            self.phi     = lambda r: np.exp(-r**2)
            self.dxphi   = lambda r,v: -2*v*np.exp(-r**2)[:,:,np.newaxis]
            self.dyphi   = lambda r,v: 2*v*np.exp(-r**2)[:,:,np.newaxis]
            self.dxdyphi = lambda r,v: 2*np.exp(-r**2)[:,:,np.newaxis,np.newaxis]*np.eye(self.netDim[0])[np.newaxis,np.newaxis,:,:] \
                                    -4*np.exp(-r**2)[:,:,np.newaxis,np.newaxis]*v[:,:,np.newaxis,:]*v[:,:,:,np.newaxis]

        else:
            raise Exception("Unknown activation function (MyKernel/init)")

        self.W1 = np.zeros(self.netMemory[0])
        self.W2 = np.zeros(n+1)

        self.K = np.zeros(((n+1)*simplexMemory,(n+1)*simplexMemory))
        self.J = np.zeros(((n+1)*simplexMemory,n*simplexMemory))

        self.V = np.ones(((n+1)*simplexMemory,n+1))
        self.D = np.zeros((n*simplexMemory,n+1))

        self.newTrainInput = np.array([0, 0],dtype=int)

    # Transform tensor to higher/lower dimensional tensor representation
    def flatten(self, tensor, outputDim=2):
        n = self.netDim[0]
        d1, d2 = np.shape(tensor)[:2]
        if len(np.shape(tensor)) == 2:
            if outputDim == 3:
                if d2%n != 0:
                    raise Exception("Incorrect input for transformation of tensor (MyRadialBasis/flatten)")
                d2 = d2//n
                return np.reshape(tensor,(d1,d2,n))

            elif outputDim == 4:
                if d1%n != 0 or d2%n != 0:
                    raise Exception("Incorrect input for transformation of tensor (MyRadialBasis/flatten)")
                d1 = d1//n
                d2 = d2//n
                return np.transpose(np.reshape(np.transpose(np.reshape(tensor,(d1*n,d2,n)),axes=[1,2,0]),(d2,n,d1,n)),axes=[2,0,3,1])
        
        elif len(np.shape(tensor)) == 3:
            if outputDim == 2:
                return np.reshape(tensor,(d1,d2*n))
        
        elif len(np.shape(tensor)) == 4:
            if outputDim == 2:
                return np.reshape(np.transpose(np.reshape(np.transpose(tensor,axes=[1,3,0,2]),(d2,n,d1*n)),axes=[2,0,1]),((d1*n,d2*n)))
            elif outputDim == 3:
                return np.reshape(np.transpose(tensor,axes=[0,2,1,3]),(d1,n,d2*n))
        
        raise Exception("Incorrect input for transformation of tensor (MyKernel/flatten)")

    # Evaluate the value of the current model
    def __call__(self, x):
        if len(np.shape(x)) == 1:
            x = x[np.newaxis]
        
        r1 = np.linalg.norm(x[:,np.newaxis,:] - self.trainInput[np.newaxis,:self.trainLength[0],:],axis=-1)
        
        val = np.matmul(self.phi(r1), self.W1[:self.trainLength[0]])
        val += np.matmul(x, self.W2[1:]) + self.W2[0]

        return np.squeeze(val,axis=0) if np.shape(x)[0] == 1 else val

    # Evaluate the gradient (with respect to input) of the current model
    def grad(self, x):
        if len(np.shape(x)) == 1:
            x = x[np.newaxis]
        
        v1 = x[:,np.newaxis,:] - self.trainInput[np.newaxis,:self.trainLength[0],:]
        r1 = np.linalg.norm(v1,axis=-1)

        val = np.matmul(np.transpose(self.dxphi(r1,v1),axes=[0,2,1]), self.W1[:self.trainLength[0]])
        val += self.W2[1:]

        return np.squeeze(val,axis=0) if np.shape(x)[0] == 1 else val

    # Update the matrices with the new training data
    def updateLinearSystem(self):
        n = self.netDim[0]
        tl = self.trainLength

        if self.trainLast[0] >= self.newTrainInput[0]:
            newIdxVal = np.arange(self.newTrainInput[0],self.trainLast[0])
        else:
            newIdxVal = np.arange(self.newTrainInput[0],tl[0]+self.trainLast[0]) % tl[0]
        
        if self.trainLast[1] >= self.newTrainInput[1]:
            newIdxDerivative = np.arange(self.newTrainInput[1],self.trainLast[1])
        else:
            newIdxDerivative = np.arange(self.newTrainInput[1],tl[1]+self.trainLast[1]) % tl[1]

        v1 = self.trainInput[newIdxVal,np.newaxis,:] - self.trainInput[np.newaxis,:tl[0],:]
        r1 = np.linalg.norm(v1,axis=-1)

        v2 = self.trainInput[:tl[0],np.newaxis,:] - self.trainInputDerivative[np.newaxis,newIdxDerivative,:]
        r2 = np.linalg.norm(v2,axis=-1)

        v3 = self.trainInput[newIdxVal,np.newaxis,:] - self.trainInputDerivative[np.newaxis,:tl[1],:]
        r3 = np.linalg.norm(v3,axis=-1)

        self.K[newIdxVal,:tl[0]] = self.phi(r1)
        self.K[:tl[0],newIdxVal] = self.K[newIdxVal,:tl[0]].T
        
        # V is initialized with 1 entries
        self.V[newIdxVal,1:] = self.trainInput[newIdxVal,:]

        if tl[1] > 0:
            newIdxDerivativeFlatten = np.reshape(n*newIdxDerivative[:,np.newaxis]+np.arange(n)[np.newaxis,:],(-1))
            self.J[:tl[0],newIdxDerivativeFlatten] = self.flatten(self.dyphi(r2,v2),outputDim=2)
            self.J[newIdxVal,:n*tl[1]] = self.flatten(self.dyphi(r3,v3),outputDim=2)
    
            self.D[n*newIdxDerivative[:,np.newaxis]+np.arange(n)[np.newaxis,:],1:] = np.eye(n)

        self.newTrainInput[0] = self.trainLast[0]
        self.newTrainInput[1] = self.trainLast[1]

    # Train the neural network on the current set of points
    def train(self, rtolTraining):
        n = self.netDim[0]
        tl = self.trainLength
        self.updateLinearSystem()

        if tl[1] > 0:
            A = np.block([[self.K[:tl[0],:tl[0]],     self.V[:tl[0]:]     ],
                          [self.J[:tl[0],:n*tl[1]].T, self.D[:n*tl[1],:]  ]])
            b = np.block([self.trainOutput[:tl[0]], np.reshape(self.trainOutputDerivative[:tl[1],:],(-1))])
        
        else:
            A = np.block([[self.K[:tl[0],:tl[0]], self.V[:tl[0]:]  ]])
            b = np.block([self.trainOutput[:tl[0]]])
            
        try:
            fullWeight = np.linalg.lstsq(A,b,rcond=None)[0]
            self.W1[:tl[0]] = fullWeight[:tl[0]]
            self.W2 = fullWeight[tl[0]:]
        except:
            print("Error SVD didn't converge so keep the same model")

        return 0

