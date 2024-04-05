import numpy as np
#from decisiontree import DecisionTree as DecisionTree


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

    def __init__(self, n_trees=25, n_features='sqrt', n_split=None, max_depth=1000, size_allowed=1):
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
            np.random.seed()  # Seed for randomness in each tree
            temp_clf = DecisionTree(n_features=self.n_features,
                                    n_split=self.n_split,
                                    max_depth=self.max_depth,
                                    size_allowed=self.size_allowed)
            temp_clf.fit(X, y)
            self.trees.append(temp_clf)

        return self

    def ind_predict(self, inp):
        """
            Predict the most likely class label of one test instance based on its feature vector x.

            TODO: 4. Modify the following code to predict using each Decision Tree.
        """
        results = np.array([tree.ind_predict(inp) for tree in self.trees])

        """
            TODO: 5. Modify the following code to find the correct prediction use majority rule.
                     If there is a tie, use random choice to select one of them.
        """
        labels, counts = np.unique(results, return_counts=True)

        return labels[np.argmax(counts)]

    def predict_all(self, inp):
        """
            X is a matrix or 2-D numpy array, represnting testing instances.
            Each testing instance is a feature vector.

            Return the predictions of all instances in a list.

            TODO: 6. Revise the following for-loop to call ind_predict to get predictions
        """
        return np.array([self.ind_predict(vect) for vect in inp])
