import numpy as np
from TreeLearner import TreeLearner

class RTLearner(TreeLearner):
    """
        Random Decision Tree Regressor class
        Superclass: TreeLearner

        randomly chooses the split column
    """

    def find_split(self, X, Y):
        """
            helper method for the addEvidence method
            picking random feature for split_col

            inputs:
                X: numpy array containing the features to split based on
                Y: numpy array containing the response

            outputs:
                split_col: integer representing the column of X to split on
                split_val: float representing the value to split split_col on;
                           median of the split column
        """
        split_col = np.random.randint(X.shape[1])
        return split_col, np.median(X[:,split_col])
