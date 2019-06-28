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

        inputs:
            layer_dims: list containing the number of neurons for each hidden layer
            alpha: float representing the learning rate of the model
            epochs: int representing the number of training epochs
            init_strategy: string representing the parameter initialization strategy
                           takes the value of "xavier" or "he"
            decay_rate: float representing the decay rate for learning rate decay
            mini_batch_size: int representing the size of each mini-batch
                             if mini_batch_size == num_examples, then performing batch gradient descent
            epsilon: float representing the adjustment value to avoid numerical instability (divide by 0)
            random_state: int for setting the np.random.seed to ensure reproducibility
            print_errors: boolean flag representing whether or not to print the cost during training

        output:
            None
    """
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

    def train(self, X, Y):
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
        self.costs = []
        #storing input and output layer dimensions
        m = X.shape[1]
        self.layer_dims.insert(0,X.shape[0])
        multiclass = len(np.unique(Y)) > 2
        if multiclass:
            self.layer_dims.append(len(set(Y)))  #for multiclass classification
        else:
            self.layer_dims.append(1)  #only doing binary classification
        num_complete_mb,incomp_mb_size = mini_batch_setup(m,self.mini_batch_size)
        #printing descriptions of the training architecture
        print("Multiclass Classification:",multiclass)
        print("Number of Layers",len(self.layer_dims))
        print("Number of Mini-Batches",num_complete_mb + (incomp_mb_size > 0))
        #training the NN with forward and back propagation
        alpha = self.alpha
        self.parameters = initialize_parameters(self.layer_dims, init_strategy=self.init_strategy)
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

    def plot_cost(self):
        """
            method for plotting the training costs

            input:
                None

            output:
                None
        """
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('Logloss')
        plt.xlabel('Iterations (per 100)')
        plt.title("Learning rate =" + str(self.alpha))
        plt.show()

    def predict(self,X):
        """
            method for predicting using learned parameters

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

            output:
                accuracy: float representing the accuracy of the current parameters
        """
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
                       decay_rate=0.001, mini_batch_size=X.shape[1]/5, init_strategy = "test",
                       random_state=0, print_errors=False)
    nn.train(X, Y)
    # nn.plot_cost()

    print("Prediction Accuracy for Neural Network:",np.round(nn.accuracy(X,Y),3),'%')
