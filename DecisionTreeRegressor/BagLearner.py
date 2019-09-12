import numpy as np


class BagLearner(object):
    """
        Ensemble method for bagging given regressor
        can be used to create a random forest regressor
    """
    def __init__(self, learner, kwargs = {}, bags = 20, verbose = False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

    def fit(self, X, Y):
        """
            method for training given learner using bootstrap aggregating
        """
        self.learners = []
        n = X.shape[0]
        for i in range(self.bags):
            bag_idx = np.random.choice(a=n,size=n,replace=True)
            X_bag = X[bag_idx]
            Y_bag = Y[bag_idx]
            self.learners.append(self.learner(**self.kwargs))
            self.learners[i].fit(X_bag,Y_bag)

    def predict(self, X):
        preds = np.zeros((self.bags, X.shape[0]))
        for i in range(self.bags):
            preds[i] = self.learners[i].query(X)
        return np.mean(preds,axis=0)
