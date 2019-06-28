import numpy as np
from activation import *

def forwardprop_helper(A_prev, W, b, activation):
    activation_func = activation_funcs[activation]
    #computing Z and linear_cache for backprop
    Z = np.dot(W,A_prev) + b
    linear_cache = (A_prev, W, b)
    #computing activations and activation_cache for use in backprop
    A, activation_cache = activation_func(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def forwardprop(X, parameters, L):
    """
        wrapper function for computing activations for each layer

        inputs:
            X: numpy array containing the training examples as column vectors
                X.shape(num_features , num_examples)
            parameters: dictionary containing the weights and intercepts for each layer
            L: int representing the number of layers

        outputs:
            AL: numpy array containing the activations of the output layer
            caches: list containing the linear and activation caches for each layer
    """
    caches = []
    A = X
    #computing activations for hidden layers
    for l in range(1, L):
        A_prev = A
        A, cache = forwardprop_helper(A_prev=A_prev,W=parameters['W' + str(l)],b=parameters['b' + str(l)],activation="relu")
        caches.append(cache)
    #computing activations for output layer
    AL, cache = forwardprop_helper(A_prev=A,W=parameters['W' + str(L)],b=parameters['b' + str(L)],activation="sigmoid")
    caches.append(cache)
    return AL, caches
