import numpy as np
#sigmoid method for computing activations during training
def sigmoid(Z):
    s = 1/(1 + np.exp(-1 * Z))
    return s , Z

#ReLU method for computing activations during training
def relu(Z):
    r = np.maximum(0,Z)
    return r , Z

def forwardprop_helper(A_prev, W, b, activation):
    #dictionary for selecting activation function
    activation_funcs = {'sigmoid':sigmoid, 'relu':relu}
    #finding activation function
    activation_func = activation_funcs[activation]
    #computing Z and linear_cache for backprop
    Z = np.dot(W,A_prev) + b
    linear_cache = (A_prev, W, b)
    #computing A and activation_cache for backprop
    A, activation_cache = activation_func(Z)
    #combining caches
    cache = (linear_cache, activation_cache)
    return A, cache

def forwardprop(X, parameters, L):
    #creating list for storing caches for each layer to use during backprop
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
