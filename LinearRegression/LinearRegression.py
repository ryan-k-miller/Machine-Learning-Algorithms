import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    """
        Linear Regression class with methods for fitting to a training dataset,
        predicting based on a testing dataset, and checking model assumptions

        inputs:
            fit_intercept: Boolean flag to determine whether to fit an intercept

        outputs:
            None
    """

    def __init__(self, fit_intercept = True):
        self.fit_intercept = fit_intercept

    def compute_fit_attributes(self, X, y):
        """
            helper method for creating attributes SSE, SSR, SST, R_squared, R_squared_adjusted, and residuals

            inputs:
                X: numpy array containing the predictor variables
                   shape = (n, p) where n = # of observations and p = # of predictors
                        or (1, p+1) if fit_intercept = True
                y: numpy array containing the response variable
                   shape = (n, 1) where n = # of observations

            outputs:
                None
        """
        n = y.shape[0]
        p = X.shape[1]
        y_pred = np.dot(X,self.beta_hat)
        y_bar = y.mean()
        self.SSE = np.linalg.norm(y - y_pred)**2
        self.SSR = np.linalg.norm(y_pred-y_bar)**2
        self.SST = np.linalg.norm(y-y_bar)**2
        self.R_squared = self.SSR/self.SST
        self.R_squared_adjusted = 1 - (1 - self.R_squared)*(n-1)/(n-p-1)

    def __repr__(self):
        return "Hello!"

    def fit(self, X, y):
        """
            method for determining the estimate of beta using the training dataset

            inputs:
                X: numpy array containing the predictor variables
                   shape = (n, p) where n = # of observations and p = # of predictors
                y: numpy array containing the response variable
                   shape = (n, 1) where n = # of observations

            outputs:
                beta_hat: numpy array containing the estimated coefficients
                          shape = (1, p) where p = # of predictors
                               or (1, p+1) if fit_intercept = True
        """
        #adding intercept to predictor array based on hyperparameter
        X_reg = np.append(X,np.ones((X.shape[0],1)),axis=1) if self.fit_intercept else X.copy()
        #computing beta hat
        X2_inv = np.linalg.inv(np.dot(X_reg.T,X_reg))
        self.beta_hat = np.dot(X2_inv,np.dot(X_reg.T,y))
        #computing measure of fit attributes
        self.compute_fit_attributes(X_reg, y)
        return self.beta_hat

    def predict(self, X):
        """
            method for predicting an unknown y based on the computed
            beta_hat and input X

            inputs:
                X: numpy array containing the predictor variables
                   shape = (n, p) where n = # of observations and p = # of predictors

            outputs:
                y_hat: numpy array containing the predictions
                       shape = (n, 1) where n = # of observations
        """
        #adding intercept to predictor array based on hyperparameter
        X_reg = np.append(X,np.ones((X.shape[0],1)),axis=1) if self.fit_intercept else X.copy()
        #computing y hat
        return np.dot(X_reg,self.beta_hat)


    def plot_fitted(self, X, y):
        """
            method for plotting the predictions against the true values
            to check for homoscedasticity

            inputs:
                X: numpy array containing the predictor variables
                   shape = (n, p) where n = # of observations and p = # of predictors
                y: numpy array containing the response variable
                   shape = (n, 1) where n = # of observations

            outputs:
                None
        """
        X_reg = np.append(X,np.ones((X.shape[0],1)),axis=1) if self.fit_intercept else X.copy()
        #plotting fitted vs true
        plt.scatter(y,np.dot(X_reg,self.beta_hat))
        #creating reference line
        plt.plot(y,y,c="black")
        plt.ylabel("True Values")
        plt.xlabel("Predicted Values")
        plt.title("True vs Fitted Values")
        plt.show()



if __name__ == "__main__":
    from sklearn import datasets, linear_model
    X,y = datasets.load_diabetes(True)
    y = y.reshape((-1,1))
    lr = LinearRegression()
    lr.fit(X,y)
    print("My Implementation's R-Squared",lr.R_squared)

    lr_sk = linear_model.LinearRegression()
    lr_sk.fit(X,y)
    print("Sklearn's R-Squared",lr_sk.score(X,y))
