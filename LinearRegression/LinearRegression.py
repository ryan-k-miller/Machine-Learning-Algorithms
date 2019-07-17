import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import probplot
import seaborn as sns

class Diagnostics:
    """
        class for checking the model fit and assumptions for Linear Regression

        inputs:
            None
        outputs:
            None
    """

    #reference for checking assumptions: https://data.library.virginia.edu/diagnostic-plots/
    #Stat 420 Book: https://daviddalpiaz.github.io/appliedstats/


    def __init__(self):
      self.SSE = None
      self.SSR = None
      self.SST = None
      self.R_squared = None
      self.R_squared_adjusted = None
      self.MSE = None

    def print_fit_attributes(self):
        """
            method for printing SSE, SSR, SST, R_squared, R_squared_adjusted, and MSE

            inputs:
                None

            outputs:
                None
        """
        print("SSE:",self.SSE)
        print("SSR:",self.SSR)
        print("SST:",self.SST)
        print("R-Squared",self.R_squared)
        print("Adjusted R-Squared:",self.R_squared_adjusted)
        print("MSE:",self.MSE)

    def residuals_fitted_plot(self):
        """
            method for plotting the predictions against the residuals to check
            if there is a linear relationship between the response and predictors

            inputs:
                None

            outputs:
                None
        """
        #plotting fitted vs residuals
        sns.residplot(np.squeeze(self.fitted),np.squeeze(self.residuals),lowess=True,line_kws={'color': 'red'})
        #creating plot labels
        plt.ylabel("Residuals")
        plt.xlabel("Fitted")
        plt.title("Residuals vs Fitted Values")
        plt.show()

    def qq_plot(self):
        """
            method for creating a QQ plot to check for normality of errors

            inputs:
                None

            outputs:
                None
        """
        norm_res = self.residuals/np.linalg.norm(self.residuals)
        plt.title("Q-Q Plot")
        probplot(x=np.squeeze(norm_res),dist='norm', plot=plt)
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Standardized Residuals")
        plt.show()

    def scale_location_plot(self):
        """
            method for creating a Scale-Location plot to check for Homoscedasticity

            inputs:
                None

            outputs:
                None
        """
        #plotting fitted vs sqrt of normalized residuals
        norm_res = self.residuals/np.linalg.norm(self.residuals)
        sqrt_res = np.sqrt(np.abs(norm_res))
        sns.residplot(np.squeeze(self.fitted),np.squeeze(sqrt_res),lowess=True,line_kws={'color': 'red'})
        #creating plot labels
        plt.ylabel("sqrt(Standardized Residuals)")
        plt.xlabel("Fitted")
        plt.title("Scale-Location Plot")
        plt.show()


class LinearRegression(Diagnostics):
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
            helper method for creating attributes SSE, SSR, SST, R_squared, R_squared_adjusted, and MSE

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
        self.MSE = self.SSE/n
        self.residuals = y - y_pred
        self.fitted = y_pred

    def __repr__(self):
        desc = self.__doc__
        return desc

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

    lr.qq_plot()
