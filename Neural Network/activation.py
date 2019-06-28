#importing dependencies
import numpy as np

#sigmoid method for computing activations during training
def sigmoid(Z):
    s = 1/(1 + np.exp(-1 * Z))
    return s , Z

#ReLU method for computing activations during training
def relu(Z):
    r = np.maximum(0,Z)
    return r , Z

#Softmax method for computing activations for output layer in multiclass classification
def softmax(Z):
    exps = np.exp(Z - Z.max())
    sum_exps = np.sum(exps)
    s = exps/sum_exps
    return s , Z

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

#defining a dictionary containing the activation functions for forward prop and backprop
activation_funcs = {'sigmoid':sigmoid, 'relu':relu}
gradient_funcs = {'sigmoid':sigmoid_gradient, 'relu':relu_gradient}
