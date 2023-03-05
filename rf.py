import numpy as np
from dtree import *
from sklearn.metrics import accuracy_score


def bootstrap(X, y, size):
    sample_size = int((len(X)/3)*2)
    idxs = np.random.choice(range(0, len(X)), size=sample_size)
    oob_idxs = [idx for idx in range(len(X)) if idx not in idxs]
    X_bootstraped = X[idxs]
    y_bootstraped = y[idxs] 
    return X_bootstraped, y_bootstraped, oob_idxs


class RandomForest621:
    def __init__(self, n_estimators=10, d_tree=None, min_samples_leaf=3):
        self.n_estimators = n_estimators
        self.d_tree = d_tree
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.
        """
        for n in range(self.n_estimators):
            X_bootstrap, y_bootstrap, oob_idxs = bootstrap(X, y, size=len(X))
            my_rf_d_tree = self.d_tree(self.min_samples_leaf, oob_idxs=oob_idxs)
            my_rf_d_tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(my_rf_d_tree)

class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3):
        super().__init__(n_estimators, d_tree=RegressionTree621, min_samples_leaf=min_samples_leaf)
        self.trees = list()
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features


    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of samples in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        # gets the set of leaves reached by x
        leaves = [tree.root.predict(X_test) for tree in self.trees]
        nobs = sum([len(leaf) for leaf in leaves])
        ysum = sum([sum(leaf) for leaf in leaves])
        return (1/nobs)*ysum

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        y_predict = [self.predict(x_test) for x_test in X_test]
        r2 = r2_score(y_test, y_predict)
        return r2

class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3):
        super().__init__(n_estimators, d_tree=ClassifierTree621, min_samples_leaf=min_samples_leaf)
        self.trees = list()
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def predict(self, X_test) -> np.ndarray:
        voting_dict = {}
        for tree in self.trees:
            tree_leaf = tree.root.predict(X_test)

            for y_hat in tree_leaf:
                y_hat = y_hat[0]
                if y_hat not in voting_dict.keys():
                    voting_dict[y_hat] = 0
                voting_dict[y_hat] += 1
        return max(voting_dict, key=voting_dict.get)


    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        y_predict = [self.predict(x_test) for x_test in X_test]
        accuracy_score_result = accuracy_score(y_test, y_predict)
        return accuracy_score_result
