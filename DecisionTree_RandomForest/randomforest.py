import numpy as np
# import pandas as pd
from decisiontree import DecisionTree as DecisionTree


class RandomForest():

    """

    RandomForest Classifier

    Attributes:
        n_trees: Number of trees.
        trees: List store each individule tree
        n_features: Number of features to use during building each individule tree.
        n_split: Number of split for each feature.
        max_depth: Max depth allowed for the tree
        size_allowed : Min_size split, smallest size allowed for split

    """

    def __init__(self, n_trees=10, n_features='sqrt', n_split='sqrt', max_depth=0, size_allowed=1):
        """
            Initilize all Attributes.

            TODO: 1. Intialize n_trees, n_features, n_split, max_depth, size_allowed.
        """
        self.n_trees = n_trees
        self.trees = []
        self.n_features = n_features
        self.n_split = n_split
        self.max_depth = max_depth
        self.size_allowed = size_allowed

    def fit(self, X, y):
        """
            The fit function fits the Random Forest model based on the training data.

            X_train is a matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.

            y_train contains the corresponding labels. There might be multiple (i.e., > 2) classes.


            TODO: 2. Modify the following for loop to create n_trees tress. by calling DecisionTree we created.
                     Pass in all the attributes.
        """
        for _ in range(self.n_trees):
            clf = DecisionTree(n_features=self.n_features,
                               n_split=self.n_split,
                               max_depth=self.max_depth,
                               size_allowed=self.size_allowed)
            if isinstance(self.n_features, str) and self.n_features == 'sqrt':
                n_features_to_consider = int(np.sqrt(X.shape[1]))
            else:
                n_features_to_consider = self.n_features

            X_subset = X[:, np.random.choice(
                X.shape[1], n_features_to_consider, replace=False)]

            clf.fit(X_subset, y)

            self.trees.append(clf)

        return self

    def ind_predict(self, inp):
        """
            Predict the most likely class label of one test instance based on its feature vector x.

            TODO: 4. Modify the following code to predict using each Decision Tree.
        """
        result = []
        for tree in self.trees:
            result.append(tree.predict([inp])[0])

        """
            TODO: 5. Modify the following code to find the correct prediction use majority rule.
                     If there is a tie, use random choice to select one of them.
        """
        vote_counts = {}
        for prediction in result:
            vote_counts[prediction] = vote_counts.get(prediction, 0) + 1

        labels = max(vote_counts, key=vote_counts.get)

        return labels

    def predict_all(self, inp):
        """
            X is a matrix or 2-D numpy array, represnting testing instances.
            Each testing instance is a feature vector.

            Return the predictions of all instances in a list.

            TODO: 6. Revise the following for-loop to call ind_predict to get predictions
        """
        result = []
        # inp = np.array(inp)
        for i in range(len(inp)):
            result.append(self.ind_predict(inp[i]))
        return result
