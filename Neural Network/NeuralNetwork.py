#importing dependencies
import numpy as np
from forwardprop import *
from backprop import *
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pandas as pd

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
        #checking

    #computing logloss of current output layer activations to track progress of training
    def cost(self, AL, Y):
        m = Y.shape[1]
        cost = (-1/m) * np.sum( Y*np.log(AL) + (1-Y)*np.log(1-AL) )
        cost = np.squeeze(cost)
        return cost

    #method to train the model using the inputted features and response var
    def train(self, X, Y):
        #initializing lists for tracking training cost ovre training iterations
        costs = []
        #storing layer dimensions and number of layers
        self.layer_dims.insert(0,X.shape[0])
        #training the NN with forward and back propagation
        self.parameters = self.initialize_parameters(self.layer_dims)
        for i in range(0, self.max_iter):
            AL, caches = forwardprop(X, self.parameters, self.L)
            grads = backprop(AL, Y, caches, self.L)
            self.parameters = self.update_parameters(self.parameters, grads)
            #storing and outputing logloss for every 100 iterations
            if i % 100 == 0:
                costs.append(self.cost(AL, Y))
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
        pred_prob,_ = forwardprop(X, self.parameters, self.L)
        pred = (pred_prob)
        return pred

    #finds the prediction accuracy of the current weights and intercept
    def accuracy(self, X, Y):
        pred = self.predict(X)
        return 100*np.mean(np.round(pred) == Y)


if __name__ == "__main__":
    #testing NeuralNetwork class using Diabetes dataset
    data = pd.read_csv("../../../Coding/diabetes.csv",header=0)
    #shaping the data so the examples are stored in column vectors
    X = np.array(data.iloc[:,:-1]).T
    Y = data.iloc[:,-1].ravel().reshape((1,-1))

    #initializing, training, and evaluating the nn
    nn = NeuralNetwork(alpha=0.01,max_iter=500,layer_dims=[20, 20, 10, 1],random_state=1)
    nn.train(X, Y)
    nn.plot_cost()

    print("Prediction Accuracy for Neural Network:",np.round(nn.accuracy(X,Y),3),'%')
