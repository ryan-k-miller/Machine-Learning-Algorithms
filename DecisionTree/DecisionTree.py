from scipy import stats
import numpy as np

#adding location of helper modules to path when run in the jupyter notebook
if __name__ != "__main__":
    import os
    import sys
    sys.path.append(os.getcwd() + "/DecisionTree/")
from helper import *


class DecisionTree(object):
    def __init__(self,depth=0):
        # Initializing the tree as a dictionary with a depth of 0
        self.tree = {'depth':depth}
        pass

    #recursively training the decision tree using the given data
    def learn(self, X, y, max_depth=15):
        X_arr = np.array(X)
        #checking stopping conditions (y only has one class or depth is >= given max_depth)
        if len(np.unique(y)) == 1 or self.tree['depth'] >= max_depth:
            self.tree['is_leaf'] = True
            self.tree['class'] = stats.mode(y).mode
        else:
            self.tree['is_leaf'] = False
            #for each attribute, finding split value that gives max information gain
            info_gain_list = []
            split_val_list = []

            #looping over all columns of X
            for i in range(X_arr.shape[1]):
                #using mode of categorical values for split
                if type(X_arr[0,i]) is str:
                    split_val = stats.mode(X_arr[:,i]).mode[0]
                #using mean of continuous variables for split
                else:
                    split_val = np.mean(X_arr[:,i])
                #splitting X based on split val and current attribute
                (X_left, X_right, y_left, y_right) = partition_classes(X, y, i, split_val)
                #storing current information gain and split val
                info_gain_list.append(information_gain(y,[y_left,y_right]))
                split_val_list.append(split_val)

            #selecting split attribute that gives max information gain
            split_attribute = np.argmax(info_gain_list)
            split_val = split_val_list[split_attribute]

            #splitting data based on best split attribute
            (X_left, X_right, y_left, y_right) = partition_classes(X, y, split_attribute, split_val)

            #storing split attribute and value
            self.tree['split_attribute'] = split_attribute
            self.tree['split_val'] = split_val_list[split_attribute]

            #recursively training child nodes
            self.tree['left'] = DecisionTree(depth=self.tree['depth']+1)
            self.tree['left'].learn(X_left,y_left,max_depth)
            self.tree['right'] = DecisionTree(depth=self.tree['depth']+1)
            self.tree['right'].learn(X_right,y_right,max_depth)

    #classifying one observation using trained tree
    def classify_one(self, record):
        if self.tree['is_leaf'] == True:
            return self.tree['class'][0]
        else:
            #splitting by categorical split_val
            if type(self.tree['split_val']) is str:
                #deciding split direction
                if record[self.tree['split_attribute']] == self.tree['split_val']:
                    return self.tree['left'].classify_one(record)
                else:
                    return self.tree['right'].classify_one(record)
            #splitting by continuous split_val
            else:
                #deciding split direction
                if record[self.tree['split_attribute']] <= self.tree['split_val']:
                    return self.tree['left'].classify_one(record)
                else:
                    return self.tree['right'].classify_one(record)

    #classifying a set of observations
    def classify(self,records):
        return [self.classify_one(i)[0] for i in records]

    #scoring the decision tree based on given test features and labels
    def score(self,X,y):
        y_pred = np.array(self.classify(X)).reshape((-1))
        return np.mean(y_pred==y.reshape((-1)))
