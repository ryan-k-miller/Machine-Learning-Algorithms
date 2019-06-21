import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


class RidgeRegression:
    """
        class for training, testing, and predicting using
        Ridge Regression on numerical data
    """
    def __init__(self, alpha, t = 0.01, max_iter = 100, use_intercept = True, standardize = True):
        self.alpha = alpha
        self.t = t
        self.max_iter = max_iter
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
        return X_std

    #method for computing the loss function with l2 regularization
    def cost(self, X, y):
        return np.linalg.norm(y - np.dot(X,w))**2 + self.alpha*np.linalg.norm(w)**2

    #method for computing the gradient of the loss function
    #with respect to w
    def grad_w(self, X, y, w, b):
        y_hat = np.dot(X,w) + b
        return np.dot(X.T, y - y_hat) + 2*self.alpha*w

    #method for computing the gradient of the loss function
    #with respect to b
    def grad_b(self, X, y, w, b):
        y_hat = np.dot(X,w) + b
        return 2*(y - y_hat)

    #method to train the model using the inputted features and response var
    def train(self, X, y, random_state = 0):
        #initializing the weights, intercept, and cost list vars
        np.random.seed(random_state)
        w = np.random.rand(X.shape[1],1)
        b = np.random.randn(X.shape[0],1)
        J = []
        #preprocessing X based on hyper-parameter standardize
        X_std = self.preprocess(X)
        #finding number of rows
        m = X_std.shape[0]
        #looping until convergence (given max_iter is reached)
        #using tqdm to show loop progress
        for i in tqdm(range(self.max_iter)):
            #calculating current predictions
            Z = np.dot(X_std,w) + b
            #calculating gradients
            dw = self.grad_w(X_std, y, w, b)
            db = self.grad_b(X_std, y, w, b)
            #updating weights and intercept
            w -= self.t*dw
            b -= self.t*db
            #appending current cost
            J.append(self.cost(X_std,y))
            #exiting loop upon convergence
            if len(J) > 1:
                if J[-1] > J[-2]:
                    break
        self.w = w
        self.b = b
        self.costs = J

    #finds the prediction accuracy of the current weights and intercept
    def mean_squared_error(self, X, y):
        #preprocessing X based on hyper-parameter standardize
        X_std = self.preprocess(X)
        Z = np.dot(X_std,self.w) + self.b
        return (1/X.shape[0]) * np.linalg.norm(Z - y.reshape(Z.shape))**2



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
