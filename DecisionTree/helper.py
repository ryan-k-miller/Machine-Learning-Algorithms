from scipy import stats
import numpy as np

#computes entropy for information gain
def entropy(class_y):
    unique_vals = list(np.unique(class_y))
    entropy = 0
    for val in unique_vals:
        prob = sum([1 for i in class_y if i == val])/len(class_y)
        entropy += prob*np.log2(prob)
    return -entropy

#partitioning classes based on given split value and variable
def partition_classes(X, y, split_attribute, split_val):
    X_left = []
    X_right = []
    y_left = []
    y_right = []

    #splitting data when split_val is categorical
    if type(split_val) is str:
        for i in range(len(X)):
            if X[i][split_attribute] == split_val:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])

    #splitting data when split_val is continuous
    else:
        for i in range(len(X)):
            if X[i][split_attribute] <= split_val:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])

    return (X_left, X_right, y_left, y_right)

#computing information gained from the current split
def information_gain(previous_y, current_y):
    #calculating previous entropy
    entropy_prev = entropy(previous_y)

    #calculating current entropy given split
    entropy_cur = 0
    total_len = sum(len(i) for i in current_y)
    for i in current_y:
        entropy_cur += entropy(i)*(len(i)/total_len)

    return entropy_prev - entropy_cur
