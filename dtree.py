import numpy as np
import statistics
from scipy.stats import mode
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from dtree import *

def gini(y):
    "Return the gini impurity score for values in y"
    _, counts = np.unique(y, return_counts=True)
    if y is None:
        return 0
    n = len(y)
    return 1 - np.sum( (counts / n)**2 )


class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if self is None: return None
        if self is LeafNode:
            return LeafNode.predict(x_test)
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test)
        else:
            return self.rchild.predict(x_test)

class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction
        self.y = y.astype(int)

    def __str__(self):
        return str(self.y)

    def predict(self, x_test):
        # return prediction
        return self.y

class RFdtrees:
    def __init__(self, min_samples_leaf=1, max_features=0.3, loss=None, oob_idxs=None):
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.loss = loss
        self.oob_idxs = oob_idxs 

    def RFbestsplit(self, X, y):
        y = y.reshape(-1, 1)
        X_y = np.hstack([X, y])
        sampled_features = np.random.choice(X.shape[1], int(X.shape[1] * self.max_features))
        # choosing k possible split to increase efficiency and generality
        k = 11
        best = {'feature': -1, 'split': -1, 'loss': self.loss(y)}
        for feature in sampled_features:
            possible_splits = np.unique(X[:, feature])
            k_selected_split = np.random.choice(possible_splits, k)
            for split in k_selected_split:
                # evaluate y values in the left and right split
                lefty = X_y[X_y[:, feature] <= split][:, -1]
                righty = X_y[X_y[:, feature] > split][:, -1]
                if len(lefty) > self.min_samples_leaf or len(righty) > self.min_samples_leaf:
                    weighted_avg_loss = (len(lefty) * self.loss(lefty) + len(righty) * self.loss(righty)) / len(y)
                    if weighted_avg_loss == 0:
                        return feature, split
                    if weighted_avg_loss < best['loss']:
                        best['loss'] = weighted_avg_loss
                        best['feature'] = feature
                        best['split'] = split
        # return the feature and split that gives the bestsplit to reduce varriance/ has the purest y
        return best['feature'], best['split']

    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for either a classifier or regressor.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressors predict the average y
        for samples in that leaf.  
              
        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)
        
    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621. create_leaf() depending
        on the type of self.
        
        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.
        (Make sure to call fit_() not fit() recursively.)
        """
        y = y.reshape(-1, 1)
        X_y= np.hstack([X, y])

        # return a leaf node if reached min_samples_leaf, purest possible y, or no more split values
        if len(y) <= self.min_samples_leaf or len(np.unique(y)) == 1 or np.unique(X, axis=0).shape[0] == 1:
            return self.create_leaf(y)
        feature, split = self.RFbestsplit(X, y)

        y_right = X_y[X_y[:, feature] > split][:, -1]
        y_left = X_y[X_y[:, feature] <= split][:, -1]
        # if no better split
        if feature == -1 or len(y_left) == 0 or len(y_right) == 0:
            return self.create_leaf(y)

        lchild = self.fit_(X_y[X_y[:,feature] <= split][:,:-1], y_left)
        rchild = self.fit_(X_y[X_y[:,feature] > split][:,:-1], y_right)
        return DecisionNode(feature, split, lchild, rchild)

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        return self.root.predict(X_test)

class RegressionTree621(RFdtrees):
    def __init__(self, min_samples_leaf=1, oob_idxs=None):
        super().__init__(min_samples_leaf, loss=np.std,  oob_idxs=oob_idxs)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        y_predict = [self.predict(x_test) for x_test in X_test]
        r2  = r2_score(y_test, y_predict)
        return r2

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))

class ClassifierTree621(RFdtrees):
    def __init__(self, min_samples_leaf=1,  oob_idxs=None):
        super().__init__(min_samples_leaf, loss=gini, oob_idxs=oob_idxs)

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        y_predict = [self.predict(x_test) for x_test in X_test]
        accuracy_score_result = accuracy_score(y_test, y_predict)
        return accuracy_score_result

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        y_values = y.reshape(y.shape[0],)
        # creates a LeafNode with mode(y)
        return LeafNode(y, mode(y)) 

