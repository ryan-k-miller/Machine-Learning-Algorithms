import numpy as np
from activation import *

#helper function for performing backprop
def backprop_helper(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m) * np.dot(dZ,A_prev.T)
    db = (1/m) * np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db

#function for performing backprop for output layer
def backprop_output(AL, Y, cache):
        linear_cache, activation_cache = cache
        dAL = np.divide(Y.astype(np.float), AL,out=np.zeros_like(AL), where=AL!=0)
        dAL -= np.divide(1 - Y.astype(np.float), 1 - AL,out=np.zeros_like(AL), where=(1-AL)!=0)
        dAL = dAL * -1
        dZ = sigmoid_gradient(dAL,activation_cache)
        return backprop_helper(dZ,linear_cache)

#function for performing backprop for hidden layers
def backprop_hidden(dA, cache):
    linear_cache, activation_cache = cache
    dZ = relu_gradient(dA,activation_cache)
    return backprop_helper(dZ,linear_cache)

def backprop(AL, Y, caches, L):
    """
        wrapper function for computing gradients of the cost function for each layer

        inputs:
            AL: numpy array containing the activations of the output layer
            Y: numpy array containing the training labels as column vectors
                if multiclass classification: Y.shape == (num_classes , num_examples)
                if binary classification: Y.shape == (1 , num_examples)
            caches: list containing the forward prop caches for each layer
            L: int representing the number of layers

        output:
            grads: dictionary containing the gradients for each layer
    """
    grads = {}
    current_cache = caches[-1]
    #finding the gradients of the parameters for the output layer
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = backprop_output(AL=AL,Y=Y,cache=current_cache)
    #finding the gradients of the parameters for the hidden layers
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l+1)], grads["db" + str(l+1)] = backprop_hidden(dA=grads['dA'+str(l+1)],cache=current_cache)
    return grads
