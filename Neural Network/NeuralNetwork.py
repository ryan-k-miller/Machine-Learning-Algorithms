#importing dependencies
import numpy as np
from forwardprop import *
from backprop import *
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
        this class is for training, predicting, and evaluating an L-depth
        Neural Network for binary classification using linear ReLU hidden nodes
        and a linear Sigmoid output node
    """
    #initializing object and defining hyper-parameters for the model
    def __init__(self, layer_dims = [4,4,4,1], alpha=0.01, max_iter=1000, random_state=0, print_errors=True):
        self.layer_dims = layer_dims
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.print_errors = print_errors
        self.L = len(layer_dims)

    #computing logloss of current output layer activations to track progress of training
    def cost(self, AL, Y):
        m = Y.shape[1]
        cost = (-1/m) * np.sum( Y*np.log(AL) + (1-Y)*np.log(1-AL) )
        cost = np.squeeze(cost)
        return cost

    def initialize_parameters(self,layer_dims):
        np.random.seed(3)
        parameters = {}
        for l in range(1, len(layer_dims)):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        return parameters

    #method for updating model parameters
    def update_parameters(self, parameters, grads):
        #updating parameters for each layer
        for l in range(1,self.L+1):
            parameters["W" + str(l)] -= self.alpha*grads["dW" + str(l)]
            parameters["b" + str(l)] -= self.alpha*grads["db" + str(l)]
            # print(l,'dW nonzero count:',np.count_nonzero(grads["dW" + str(l)]))
        return parameters

    #method to train the model using the inputted features and response var
    def train(self, X, Y):
        # keep track of training cost and accuracy
        costs = []
        acc = []
        #storing layer dimensions (number of observations, number of hidden nodes,
        #number of output nodes) and number of layers
        self.layer_dims.insert(0,X.shape[0])
        print("train L:",self.L)
        #initializing parameters
        self.parameters = self.initialize_parameters(self.layer_dims)
        #looping until convergence (max_iter reached)
        for i in range(0, self.max_iter):
            #performing forward propagation
            AL, caches = forwardprop(X, self.parameters, self.L)
            #performing backpropagation.
            grads = backprop(AL, Y, caches, self.L)
            #updating parameters
            self.parameters = self.update_parameters(self.parameters, grads)
            if i % 100 == 0:
                #storing training cost and accuracy
                costs.append(self.cost(AL, Y))
                # print("mean AL:",np.mean(AL))
                #printing current logloss based on hyperparameters
                if self.print_errors:
                    print ("Logloss after iteration %i: %f" %(i, costs[-1]))
        self.costs = costs

    #method for plotting cost over the training iterations
    def plot_cost(self):
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('Logloss')
        plt.xlabel('Iterations (per 1000)')
        plt.title("Learning rate =" + str(self.alpha))
        plt.show()

    #method for predicting using learned update_parameters
    def predict(self,X):
        #making predictions using current parameters
        pred_prob,_ = forwardprop(X, self.parameters, self.L)
        pred = (pred_prob)
        return pred

    #finds the prediction accuracy of the current weights and intercept
    def accuracy(self, X, Y):
        #making predictions
        pred = self.predict(X)
        return 100*np.mean(np.round(pred) == Y)
