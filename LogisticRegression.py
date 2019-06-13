#importing dependencies
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as sklr
import matplotlib.pyplot as plt

class LogisticRegression:
    """
        this class is for training, predicting, and evaluating the binary class
        Logistic Regression algorithm using Gradient Descent
    """

    #initializing object and defining hyper-parameters for the model
    def __init__(self, alpha=0.01, max_iter=100, random_state=0, standardize=True):
        self.alpha= alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.standardize = standardize

    #sigmoid method to use during training and predicting
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    #cost function to track progress of training
    def cost(self, A, y):
        return np.mean( -( y*np.log(A) + (1-y)*np.log(1-A) ) )

    #standardizing the dataset to mean of 0 and unit variance when standardize=True
    #to allow for faster computation and avoid overflow errors
    def preprocess(self, X):
        if self.standardize == True:
            ss = StandardScaler()
            X_std = ss.fit_transform(X)
        else:
            X_std = X
        return X_std

    #determines the best step size for the current iteration
    def adaptive_step_size(self, X_std, w, b, dw, db, t, y):
        itr = 0
        #calculating current and new A
        Z = np.dot(X_std,w.T) + b
        A = self.sigmoid(Z)
        Z_new = np.dot( X_std, (w - t*dw).T ) + b - t*db
        A_new = self.sigmoid(Z_new)
        #decreasing step size until cost function doesn't increase
        while self.cost(A,y) < self.cost(A_new,y) and itr < 5:
            #updating step size and iteration counter
            t = t*0.1
            itr += 1
            #calculating new A
            A = A_new
            Z_new = np.dot(X_std,(w - t*dw).T) + b - t*db
            A_new = self.sigmoid(Z_new)

        return t

    #method to train the model using the inputted features and response var
    def train(self, X, y, random_state = 0):
        #initializing the weights, intercept, and cost list vars
        np.random.seed(random_state)
        w = np.random.rand(1,X.shape[1])
        b = 0
        J = []
        t = self.alpha
        #preprocessing X based on hyper-parameter standardize
        X_std = self.preprocess(X)
        #finding number of rows
        m = X_std.shape[0]
        #looping until convergence (given max_iter is reached)
        #using tqdm to show loop progress
        for i in tqdm(range(self.max_iter)):
            #calculating current predictions
            Z = np.dot(X_std,w.T) + b
            A = self.sigmoid(Z)
            #calculating gradients
            dZ = A - y.reshape((-1,1))
            dw = (1/m)*np.dot(dZ.T,X_std)
            db = (1/m)*np.sum(dZ)
            #updating weights and intercept
            t = self.adaptive_step_size(X_std, w, b, dw, db, t, y)
            w -= t*dw
            b -= t*db
            #appending current cost
            J.append(self.cost(A,y))
            #exiting loop upon convergence
            if len(J) > 1:
                if J[-1] > J[-2]:
                    break
        self.w = w
        self.b = b
        self.J = J

    #finds the prediction accuracy of the current weights and intercept
    def accuracy(self, X, y):
        #preprocessing X based on hyper-parameter standardize
        X_std = self.preprocess(X)
        Z = np.dot(X_std,self.w.T) + self.b
        pred_prob = np.exp(Z)/(1+np.exp(Z))
        pred = [1 if i > 0.5 else 0 for i in pred_prob]
        return 100*np.mean(pred == y.reshape(-1))




#testing LogisticRegression class
data = pd.read_csv("../Data/Binary Classification/diabetes.csv",header=0)
X = data.iloc[:,:-1]
y = data.iloc[:,-1].ravel()
lr = LogisticRegression(alpha=0.1,max_iter=200,standardize=True,random_state=0)
lr.train(X, y)

print("Prediction Accuracy for Gradient Descent:",np.round(lr.accuracy(X,y),3),'%')
print(lr.w,lr.b)
print("Number of Iterations until Convergence for Gradient Descent:",len(lr.J))

#comparing to sklearn's implementation
sklearn_lr = sklr(fit_intercept=True,solver='lbfgs',max_iter=200, random_state=0)
sklearn_lr.fit(X,y)
print("Prediction Accuracy for Sklearn:",np.round(100*sklearn_lr.score(X,y),3),'%')
print(sklearn_lr.coef_,sklearn_lr.intercept_)
print("Number of Iterations until Convergence for Sklearn:",sklearn_lr.n_iter_[0])

# #plotting cost across iterations
gd_likelihoods = np.array(lr.J).reshape((-1))
plt.plot(gd_likelihoods)
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.title("Cost over Training Iterations")
plt.show()
