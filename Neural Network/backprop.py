import numpy as np
from forwardprop import sigmoid
#gradient of the sigmoid method for performing backpropagation
def sigmoid_gradient(dA, Z):
    s,_ = sigmoid(Z)
    sg  = s*(1-s)
    return dA*sg

#gradient of the ReLU method for performing backpropagation
def relu_gradient(dA, Z):
    dZ = np.array(dA,copy=True)
    dZ[Z <= 0] = 0
    return dZ

#helper method for performing backprop
def backprop_helper1(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m) * np.dot(dZ,A_prev.T)
    db = (1/m) * np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db

#helper method for performing backprop
def backprop_helper2(dA, cache, activation):
    #finding gradient activation function to use
    gradient_funcs = {'sigmoid':sigmoid_gradient, 'relu':relu_gradient}
    gradient_func = gradient_funcs[activation]
    #splitting caches for later use
    linear_cache, activation_cache = cache
    #finding gradient of Z
    dZ = gradient_func(dA,activation_cache)
    return backprop_helper1(dZ,linear_cache)

#method for performing backpropagation
def backprop(AL, Y, caches, L):
    #initializing dictionary for storing gradient values
    grads = {}
    #storing the number of observations to be used later
    m = AL.shape[1]
    #computing the derivative of the cost function with respect to AL
    dAL = -1 * (np.divide(Y.astype(np.float), AL,out=np.zeros_like(AL), where=AL!=0) - np.divide(1 - Y.astype(np.float), 1 - AL,out=np.zeros_like(AL), where=(1-AL)!=0))
    #pulling the last cache to use later
    current_cache = caches[-1]
    #finding the gradients of the parameters for the output layer
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = backprop_helper2(dA=dAL,activation="sigmoid",cache=current_cache)
    #finding the gradients of the parameters for the hidden layers
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l+1)], grads["db" + str(l+1)] = backprop_helper2(dA=grads['dA'+str(l+1)],activation="relu",cache=current_cache)
    return grads
