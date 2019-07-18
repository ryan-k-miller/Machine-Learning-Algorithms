from scipy import stats
import numpy as np
from DecisionTree.DecisionTree import entropy,partition_classes,information_gain,DecisionTree

class RandomForest(object):
    #list to contain the decision trees made during fitting
    decision_trees = []

    #the bootstrapping datasets for trees
    #bootstraps_datasets is a list of lists, where each list in bootstraps_datasets is a bootstrapped dataset.
    bootstraps_datasets = []

    #the true class labels, corresponding to records in the bootstrapping datasets
    #bootstraps_labels is a list of lists, where the 'i'th list contains the labels corresponding to records in
    #the 'i'th bootstrapped dataset.
    bootstraps_labels = []

    def _bootstrapping(self, X, y, boot_size):
        #finding random sample of indices
        np.random.seed = 0
        idxs = np.random.randint(low=0,high=len(X),size=int(boot_size*len(X)))
        #creating the bootstrapped features and labels
        samples = [list(X[i,:]) for i in idxs]
        labels = [y[i] for i in idxs]
        return (samples, labels)

    def bootstrapping(self, X, y, boot_size):
        #checking to see if bootstraps_datasets is already populated
        if len(self.bootstraps_datasets) > 0:
            return
        #initializing one bootstapped dataset for each tree
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(X, y, boot_size)
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)

    def fitting(self, X, y, num_trees = 20, max_depth = 15, boot_size = 0.20):
        #initializing decision trees
        self.num_trees = num_trees
        self.decision_trees = [DecisionTree() for i in range(num_trees)]
        #self.bootstraps_datasets = []
        #creating bootstrapped datasets
        self.bootstrapping(X,y,boot_size)
        #training the decision trees using the bootstrapped datasets
        for i in range(len(self.decision_trees)):
            self.decision_trees[i].learn(X=self.bootstraps_datasets[i],y=self.bootstraps_labels[i], max_depth=max_depth)

    def voting(self, X):
        y = []
        #looping over all observations in X
        for record in X:
            votes = []
            #looping over all bootstrapped datasets
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]
                #if the record is not in the bootstrapped dataset
                #getting the votes from the out-of-bag trees
                if list(record) not in dataset:
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify_one(record)
                    votes.append(effective_vote[0])
            counts = np.bincount(votes)

            #if the record is not an out-of-bag sample for any of the trees
            #take the majority vote of all the trees
            if len(counts) == 0:
                for i in range(len(self.bootstraps_datasets)):
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify_one(record)
                    votes.append(effective_vote)
                counts = np.bincount(votes)
                y = np.append(y, np.argmax(counts))
            else:
                y = np.append(y, np.argmax(counts))
        return y
