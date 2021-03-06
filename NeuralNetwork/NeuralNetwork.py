#importing dependencies
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pandas as pd

#adding location of helper modules to path when run in the jupyter notebook
if __name__ != "__main__":
    import os
    import sys
    sys.path.append(os.getcwd() + "/NeuralNetwork/")

from forwardprop import *
from backprop import *
from parameters import *
from mini_batch import *


class NeuralNetwork:
    """
        this class is for training, predicting, and evaluating an L-depth
        Neural Network for binary classification using linear ReLU hidden nodes
        and a linear Sigmoid output node

        inputs:
            layer_dims: list containing the number of neurons for each hidden layer
            alpha: float representing the learning rate of the model
            epochs: int representing the number of training epochs
            init_strategy: string representing the parameter initialization strategy
                           takes the value of "xavier" or "he"
            decay_rate: float representing the decay rate for learning rate decay
            mini_batch_size: int representing the size of each mini-batch
                             if mini_batch_size == num_examples, then performing batch
                             gradient descent
            epsilon: float representing the adjustment value to avoid numerical instability
                     (divide by 0)
            random_state: int for setting the np.random.seed to ensure reproducibility
            print_errors: boolean flag representing whether or not to print the cost during training
            print_iter: the number of iterations between printing the current training cost

        output: None
    """
    def __init__(self, layer_dims=[4,4,4], alpha=0.01, lmbda = 0.0,
                 init_strategy="Xavier",decay_rate=0.001, mini_batch_size=64,
                 epsilon=10**-8, random_state=0, print_errors=True, print_iter = 100):
        #validating types of inputs
        assert isinstance(layer_dims,list)
        assert isinstance(alpha,float)
        assert isinstance(lmbda,float)
        assert isinstance(init_strategy,str)
        assert isinstance(decay_rate,float)
        assert isinstance(mini_batch_size,int)
        assert isinstance(epsilon,float)
        assert isinstance(random_state,int)
        assert isinstance(print_errors,bool)
        assert isinstance(print_iter,int)

        #assigning inputs as atrtibutes of the class
        self.layer_dims = layer_dims
        self.alpha = alpha
        self.random_state = int(random_state)
        self.print_errors = print_errors
        self.L = len(layer_dims) + 1
        self.init_strategy = init_strategy.lower()
        self.decay_rate = decay_rate
        self.mini_batch_size = int(mini_batch_size)
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.print_iter = print_iter

    #computing logloss of current output layer activations to track progress of training
    def cost_binary(self, AL, Y):
        m = Y.shape[1]
        cost = -1 * np.mean( Y*np.log(AL + self.epsilon) + (1-Y)*np.log(1 - AL + self.epsilon) )
        cost = np.squeeze(cost)
        return cost

    #computing logloss of current output layer activations to track progress of training
    def cost_multi(self, AL, Y):
        cost = -1 * np.mean( np.sum(Y*np.log(AL + self.epsilon),axis=0 ) )
        cost = np.squeeze(cost)
        return cost

    def train_helper(self, X_batch, Y_batch, alpha, m):
        AL, caches = forwardprop(X_batch, self.parameters, self.L)
        grads = backprop(AL, Y_batch, caches, self.L)
        self.parameters = update_parameters(self.parameters, grads, alpha, self.L, self.lmbda, m)
        return AL

    def train(self, X, Y, epochs=100, retrain = False):
        """
            method for training the Neural Network based on the hyperparameters
            selected during initialization

            parameters and training costs are stored as attributes of the class

            inputs:
                X: numpy array containing the training examples as column vectors
                    X.shape == (num_features , num_examples)
                Y: numpy array containing the training labels as column vectors
                    if multiclass classification: Y.shape == (num_classes , num_examples)
                    if binary classification: Y.shape == (1 , num_examples)

            output:
                None
        """
        #checking to see if number of observations in X and Y match
        assert X.shape[1] == Y.shape[1]
        #checking if epochs is an int
        assert isinstance(epochs,int)

        self.costs = []
        #storing input and output layer dimensions
        self.layer_dims.insert(0,X.shape[0])
        self.num_classes = Y.shape[0]
        out_dim = self.num_classes if self.num_classes > 2 else 1
        self.layer_dims.append(out_dim)
        #determining mini batch sizes
        m = X.shape[1]
        num_complete_mb,incomp_mb_size = mini_batch_setup(m,self.mini_batch_size)
        #printing descriptions of the training architecture
        print("Shape of Training Data:",X.shape)
        print("Number of Classes:",self.num_classes)
        print("Number of Layers",len(self.layer_dims))
        print("Layer Dimensions:",self.layer_dims)
        print("Number of Mini-Batches",num_complete_mb + (incomp_mb_size > 0))
        #if retrain is True, then use current parameters
        if retrain == False:
            self.parameters = initialize_parameters(self.layer_dims, init_strategy=self.init_strategy)
        #training the NN with forward and back propagation
        for i in range(0, epochs):
            alpha = self.alpha/(1+self.decay_rate*i)
            for t in range(num_complete_mb):
                X_batch = X[:,t*self.mini_batch_size:(t+1)*self.mini_batch_size]
                Y_batch = Y[:,t*self.mini_batch_size:(t+1)*self.mini_batch_size]
                AL = self.train_helper(X_batch, Y_batch, alpha, m)
                #storing and outputing logloss for every 100 iterations
                if (i*(num_complete_mb + (incomp_mb_size > 0)) + t + 1) % self.print_iter == 0:
                    cost = self.cost_multi(AL,Y_batch) if self.num_classes > 2 else self.cost_binary(AL,Y_batch)
                    self.costs.append(cost)
                    if self.print_errors:
                        print ("Logloss after iteration %i: %f" %((i*(num_complete_mb + (incomp_mb_size > 0))) + t + 1, self.costs[-1]))
            #performing GD on incomplete mini-batch if exists
            if incomp_mb_size != 0:
                X_batch = X[:,-incomp_mb_size:]
                Y_batch = Y[:,-incomp_mb_size:]
                AL = self.train_helper(X_batch, Y_batch, alpha, m)

    def predict(self,X):
        """
            method for prediction using learned parameters

            input:
                X: numpy array containing the examples as column vectors for prediction
                    X.shape == (num_features , num_examples)
                Y: numpy array containing the training labels as column vectors
                    if multiclass classification: Y.shape == (num_classes , num_examples)
                    if binary classification: Y.shape == (1 , num_examples)

            output:
                pred_prob: numpy array containing the predicted probabilities for each example
                    if multiclass classification: pred_prob.shape == (num_classes , num_examples)
                    if binary classification: pred_prob.shape == (1 , num_examples)
        """
        pred_prob,_ = forwardprop(X, self.parameters, self.L)
        return pred_prob

    def accuracy(self, X, Y):
        """
            method for finding the accuracy of the current parameters

            input:
                X: numpy array containing the examples as column vectors for prediction
                    X.shape == (num_features , num_examples)
                Y: numpy array containing the training labels as column vectors
                    if multiclass classification: Y.shape == (num_classes , num_examples)
                    if binary classification: Y.shape == (1 , num_examples)

            output:
                accuracy: float representing the accuracy of the current parameters
        """
        pred = np.round(self.predict(X)).reshape(Y.shape)
        if self.num_classes > 2:
            labels = np.array(range(self.num_classes)).reshape((1,-1))
            pred = np.dot(labels,pred)
            Y = np.dot(labels,Y)
        return 100*np.mean(pred == Y)

    def plot_cost(self):
        """ method for plotting the training costs """
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('Logloss')
        plt.xlabel('Iterations (per %f)'%self.print_iter)
        plt.title("Learning rate =" + str(self.alpha))
        plt.show()


if __name__ == "__main__":
    #testing NeuralNetwork class using MNIST dataset
    from tensorflow.keras.datasets import mnist
    (X_train,Y_train), (X_test,Y_test) = mnist.load_data()
    X_train = X_train.astype(np.float16).reshape((-1,X_train.shape[1]*X_train.shape[2]))
    X_train = X_train.T / 255
    Y_train = pd.get_dummies(Y_train).values.T
    # X_test = test.iloc[:,1:].values
    X_test = X_test.astype(np.float16).reshape((-1,X_test.shape[1]*X_test.shape[2]))
    X_test = X_test.T / 255
    Y_test = pd.get_dummies(Y_test).values.T
    X_train.shape
    #initializing, training, and evaluating the nn
    nn = NeuralNetwork(alpha=0.1,layer_dims=[100,100,50,30,30], lmbda = 0.5,
                       decay_rate=0.3, mini_batch_size=128, init_strategy = "xavier",
                       random_state=0, print_errors=True)
    nn.train(X_train, Y_train,epochs=5)
    # nn.plot_cost()

    print("Test Accuracy for Neural Network:",np.round(nn.accuracy(X_test,Y_test),3),'%')
