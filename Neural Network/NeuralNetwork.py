#importing dependencies
import numpy as np
from forwardprop import *
from backprop import *
from parameters import *
from mini_batch import *
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
        this class is for training, predicting, and evaluating an L-depth
        Neural Network for binary classification using linear ReLU hidden nodes
        and a linear Sigmoid output node
    """
    #initializing object and defining hyper-parameters for the model
    def __init__(self, layer_dims=[4,4,4], alpha=0.01, epochs=1000,
                 init_strategy="Xavier",decay_rate=0.001, mini_batch_size=64,
                 epsilon=10**-8, random_state=0, print_errors=True):
        self.layer_dims = layer_dims
        self.alpha = alpha
        self.epochs = int(epochs)
        self.random_state = int(random_state)
        self.print_errors = print_errors
        self.L = len(layer_dims)
        self.init_strategy = init_strategy.lower()
        self.decay_rate = decay_rate
        self.mini_batch_size = int(mini_batch_size)
        self.epsilon = epsilon

    #computing logloss of current output layer activations to track progress of training
    def cost(self, AL, Y):
        m = Y.shape[1]
        cost = (-1/m) * np.sum( Y*np.log(AL + self.epsilon) + (1-Y)*np.log(1 - AL + self.epsilon) )
        cost = np.squeeze(cost)
        return cost

    #method to train the model using the inputted features and response var
    def train(self, X, Y):
        self.costs = []
        #storing input and output layer dimensions
        m = X.shape[1]
        self.layer_dims.insert(0,X.shape[0])
        self.layer_dims.append(1)  #only doing binary classification
        #self.layer_dims.append(len(set(Y)))  #for multiclass classification
        alpha = self.alpha
        num_complete_mb,incomp_mb_size = mini_batch_setup(m,self.mini_batch_size)
        print("Number of Mini-Batches",num_complete_mb + (incomp_mb_size > 0))
        #training the NN with forward and back propagation
        self.parameters = initialize_parameters(self.layer_dims)
        for i in range(0, self.epochs):
            alpha = self.alpha/(1+self.decay_rate*i)
            for t in range(num_complete_mb):
                X_batch = X[:,t*self.mini_batch_size:(t+1)*self.mini_batch_size]
                Y_batch = Y[:,t*self.mini_batch_size:(t+1)*self.mini_batch_size]
                AL, caches = forwardprop(X_batch, self.parameters, self.L)
                grads = backprop(AL, Y_batch, caches, self.L)
                self.parameters = update_parameters(self.parameters, grads, alpha, self.L)
                #storing and outputing logloss for every 100 iterations
                if (i*(num_complete_mb + (incomp_mb_size > 0)) + t + 1) % 100 == 0:
                    self.costs.append(self.cost(AL, Y_batch))
                    if self.print_errors:
                        print ("Logloss after iteration %i: %f" %((i*(num_complete_mb + (incomp_mb_size > 0))) + t + 1, self.costs[-1]))
            #performing GD on incomplete mini-batch if exists
            if incomp_mb_size != 0:
                X_batch = X[:,-incomp_mb_size:]
                Y_batch = Y[:,-incomp_mb_size:]
                AL, caches = forwardprop(X_batch, self.parameters, self.L)
                grads = backprop(AL, Y_batch, caches, self.L)
                self.parameters = update_parameters(self.parameters, grads, alpha, self.L)
                alpha = self.alpha/(1+self.decay_rate*i)

    #method for plotting cost over the training iterations
    def plot_cost(self):
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('Logloss')
        plt.xlabel('Iterations (per 100)')
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
    import pandas as pd
    #testing NeuralNetwork class using Diabetes dataset
    data = pd.read_csv("../../../Coding/diabetes.csv",header=0)
    #shaping the data so the examples are stored in column vectors
    X = np.array(data.iloc[:,:-1]).T
    Y = data.iloc[:,-1].ravel().reshape((1,-1))
    print("Shape of Training Data:",X.shape)
    #initializing, training, and evaluating the nn
    nn = NeuralNetwork(alpha=0.001,epochs=5000,layer_dims=[20, 20, 10, 10],
                       decay_rate=0.001, mini_batch_size=X.shape[1]/5, init_strategy = "Xavier",
                       random_state=0, print_errors=False)
    nn.train(X, Y)
    nn.plot_cost()

    print("Prediction Accuracy for Neural Network:",np.round(nn.accuracy(X,Y),3),'%')
