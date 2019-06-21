import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class RidgeRegression:
    """
        class for training, testing, and predicting using
        Ridge Regression on numerical data
    """
    def __init__(self, alpha, use_intercept = True, standardize = True):
        self.alpha = alpha
        self.use_intercept = use_intercept
        self.standardize = standardize

    #standardizing the dataset to mean of 0 and unit variance when standardize=True
    #to allow for faster computation and avoid overflow errors
    def preprocess(self, X):
        if self.standardize == True:
            ss = StandardScaler()
            X_std = ss.fit_transform(X)
        else:
            X_std = X
        if self.use_intercept == True:
            X_std = np.append(X_std,np.ones((X_std.shape[0],1)))
        return X_std

    #method for computing MSE + l2 regularization term
    def cost(self, X, y, w):
        y_hat = np.dot(X,w)
        return np.linalg.norm(y - y_hat)**2 + self.alpha*np.linalg.norm(w)**2

    #method for computing the gradient of the loss function
    #with respect to w
    def update_w(self, X, y):
        t1 = np.linalg.inv( np.dot(X.T,X) + self.alpha*np.eye(X.shape[1]) )
        t2 = np.dot(X.T,y)
        return np.dot(t1, t2)

    #method to train the model using the inputted features and response var
    def train(self, X, y, random_state = 0):
        #preprocessing X based on hyper-parameter standardize
        X_std = self.preprocess(X)
        #initializing the weights, intercept, and cost list vars
        np.random.seed(random_state)
        w = np.random.rand(X.shape[1],1)
        J = []
        #updating parameters
        w = self.update_w(X_std, y, b)
        b = self.update_b(X_std, y, w)
        self.w = w
        self.b = b
        self.costs = J

    #finds the prediction accuracy of the current weights and intercept
    def mean_squared_error(self, X, y):
        #preprocessing X based on hyper-parameter standardize
        X_std = self.preprocess(X)
        Z = np.dot(X_std,self.w) + self.b
        return (1/X.shape[0]) * np.linalg.norm(Z - y)**2


#reading in data, removing categorical columns, dropping rows with NAs
data = pd.read_csv("../../Coding/train.csv",header=0)
num_cols = ['LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','GrLivArea','BedroomAbvGr','TotRmsAbvGrd','Fireplaces','YrSold','SalePrice']
data = data.filter(num_cols,axis=1)
data.dropna(axis=0,inplace=True)
#splitting data into X and y
X = data.iloc[:,:-1]
y = data.iloc[:,-1].ravel().reshape((-1,1))
#training RidgeRegression model
lr = RidgeRegression(alpha=1,standardize=True)
lr.train(X, y)

print("Train MSE for Gradient Descent:",np.round(lr.mean_squared_error(X,y),10))
# print(lr.w,lr.b)
print("Number of Iterations until Convergence for Gradient Descent:",len(lr.costs))

#comparing to sklearn's implementation
sklearn_lr = Ridge(fit_intercept=True,normalize=True,solver='svd',max_iter=200, random_state=0)
sklearn_lr.fit(X,y)
print("Train MSE for Sklearn:",np.round(100*sklearn_lr.score(X,y),3))
# # print(sklearn_lr.coef_,sklearn_lr.intercept_)
# print("Number of Iterations until Convergence for Sklearn:",sklearn_lr.n_iter_[0])

# # #plotting cost across iterations
# gd_likelihoods = np.array(lr.costs).reshape((-1))
# plt.plot(gd_likelihoods)
# plt.ylabel("Cost")
# plt.xlabel("Iterations")
# plt.title("Cost over Training Iterations")
# plt.show()
