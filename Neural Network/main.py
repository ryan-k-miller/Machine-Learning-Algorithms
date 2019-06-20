import pandas as pd
import numpy as np
from NeuralNetwork import *

#testing LogisticRegression class
data = pd.read_csv("../Data/Binary Classification/diabetes.csv",header=0)
#shaping the data so the examples are stored in column vectors
X = data.iloc[:,:-1].T
Y = data.iloc[:,-1].ravel().reshape((1,-1))

#initializing, training, and evaluating the nn
nn = NeuralNetwork(alpha=0.01,max_iter=5000,layer_dims=[20, 20, 20, 20, 10, 1],random_state=1)
nn.train(X, Y)
nn.plot_cost()

print("Prediction Accuracy for Neural Network:",np.round(nn.accuracy(X,Y),3),'%')
