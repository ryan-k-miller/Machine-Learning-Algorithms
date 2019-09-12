import numpy as np
from TreeLearner import TreeLearner

class DTLearner(TreeLearner):
    """
        Decision Tree Regressor class
        Superclass: TreeLearner

        uses correlation to find the optimal split column
    """

    def find_split(self, X, Y):
        """
            helper method for the addEvidence method
            finding the feature that is most highly correlated with the response

            inputs:
                X: numpy array containing the features to split based on
                Y: numpy array containing the response

            outputs:
                split_col: integer representing the column of X to split on
                split_val: float representing the value to split split_col on;
                           median of the split column
        """
        corrs = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            #checking if column of X is constant (causes div by 0 error)
            if len(np.unique(X[:,i])) > 1:
                corrs[i] = abs(np.corrcoef(Y,X[:,i]))[0,1]
        split_col = np.argmax(corrs)
        return split_col, np.median(X[:,split_col])
