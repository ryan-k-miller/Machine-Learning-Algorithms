import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression():
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
        if self.fit_intercept == True:
            X_reg = np.append(X,np.ones((X.shape[0],1)))
        else:
            X_reg = X.copy()
        #computing beta hat
        X2_inv = np.linalg.inv(np.dot(X_reg.T,X_reg))
        self.beta_hat = np.dot(X2_inv,np.dot(X_reg.T,y))
        #computing SSE, SSR, SST, and R_squared
        y_pred = np.dot(X_reg,self.beta_hat)
        y_bar = np.mean(y)
        self.SSE = np.linalg.norm(y-y_pred)**2
        self.SSR = np.linalg.norm(y_pred-y_bar)**2
        self.SST = np.linalg.norm(y-y_bar)**2
        self.R_squared = self.SSR/self.SST
