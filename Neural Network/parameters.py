import numpy as np

def initialize_parameters(layer_dims, init_strategy = 'xavier', random_state = 0):
    """
        initializing the weights and intercepts of the neural network

        inputs:
            layers_dims: list containing the number of neurons in each layer
            init_strategy: takes either 'He' or 'Xavier'
                           chooses the initialization strategy for the weights
           random_state: value for setting the random state

        output:
            parameters: dictionary containing the initialized weights and intercepts for each layer
    """
    np.random.seed(random_state)
    parameters = {}
    if init_strategy == 'he':
        init_multiplier = 2
    elif init_strategy == 'xavier':
        init_multiplier = 1
    for l in range(1, len(layer_dims)):
        parameters['W' + str(l)] = (np.random.randn(layer_dims[l], layer_dims[l-1])) * init_multiplier/ np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
    return parameters

#method for updating model parameters for all layers
def update_parameters(parameters, grads, alpha, L):
    """
        updating parameters based on the gradients derived during backprop
        inputs:
            parameters: dictionary containing the current weights and intercepts for each layer
            grads: dictionary containing the gradients of the weights and intercepts for each layer
        output:
            parameters: dictionary containing the updated weights and intercepts for each layer
    """
    for l in range(1,L+1):
        parameters["W" + str(l)] -= alpha*grads["dW" + str(l)]
        parameters["b" + str(l)] -= alpha*grads["db" + str(l)]
    return parameters
