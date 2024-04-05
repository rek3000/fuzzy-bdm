import numpy as np
import math


class DecisionTree():
    """

    Decision Tree Classifier

    Attributes:
        root: Root Node of the tree.
        max_depth: Max depth allowed for the tree
        size_allowed : Min_size split, smallest size allowed for split
        n_features: Number of features to use during building the tree.(Random Forest)
        n_split:  Number of split for each feature. (Random Forest)

    """

    def __init__(self, max_depth=1000, size_allowed=1, n_features=None, n_split=None):
        """

            Initializations for class attributes.

            DONE TODO: 1. Modify the initialization of the attributes of the Decision Tree classifier
        """

        self.root = None
        self.max_depth = max_depth
        self.size_allowed = size_allowed
        self.n_features = n_features
        self.n_split = n_split

    class Node():
        """
            Node Class for the building the tree.

            Attribute:
                threshold: The threshold like if x1 < threshold, for spliting.
                feature: The index of feature on this current node.
                left: Pointer to the node on the left.
                right: Pointer to the node on the right.
                pure: Bool, describe if this node is pure.
                predict: Class, indicate what the most common Y on this node.

        """

        def __init__(self, threshold=None, feature=None, depth=0):
            """

                Initializations for class attributes.

                DONE TODO: 2. Modify the initialization of the attributes of the Node. (Initialize threshold and feature)
            """

            self.threshold = threshold
            self.feature = feature
            self.left = None
            self.right = None
            self.pure = False
            self.predict = None
            self.depth = depth

    def entropy(self, lst):
        """
            Function Calculate the entropy given lst.

            Attributes:
                entro: variable store entropy for each step.
                classes: all possible classes. (without repeating terms)
                counts: counts of each possible classes.
                total_counts: number of instances in this lst.

            lst is vector of labels.



            TODO: 3. Intilize attributes.
                  4. Modify and add some codes to the following for-loop
                     to compute the correct entropy.
                     (make sure count of corresponding label is not 0, think about why we need to do that.)
        """
        entro = 0
        classes, counts = np.unique(lst, return_counts=True)
        total_counts = len(lst)

        # Avoid division by zero for empty classes
        for count in counts:
            if count > 0:
                p = count / total_counts
                entro -= p * np.log2(p)

        return entro

    def information_gain(self, lst, values, threshold):
        """

            Function Calculate the information gain, by using entropy function.

            lst is vector of labels.
            values is vector of values for individule feature.
            threshold is the split threshold we want to use for calculating the entropy.


            TODO:
                5. Modify the following variable to calculate the P(left node), P(right node),
                   Conditional Entropy(left node) and Conditional Entropy(right node)
                6. Return information gain.


        """
        left_index = values <= threshold
        right_index = values > threshold

        left_entropy = self.entropy(lst[left_index])
        right_entropy = self.entropy(lst[right_index])

        p_left = len(lst[left_index]) / len(lst)
        p_right = len(lst[right_index]) / len(lst)

        information_gain = self.entropy(
            lst) - p_left * left_entropy - p_right * right_entropy
        return information_gain

    def find_rules(self, data):
        """

            Helper function to find the split rules.

            data is a matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.

            TODO: 7. Modify the following for loop, which loop through each column(feature).
                     Find the unique value of each feature, and find the mid point of each adjacent value.
                  8. Store them in a list, return that list.

        """
        n, m = data.shape
        rules = []

        for col in range(m):
            unique_values = np.unique(data[:, col])
            diff = np.diff(unique_values)

            potential_thresholds = unique_values[:-1] + diff / 2
            rules.append(potential_thresholds.tolist())

        return rules

    def next_split(self, data, label, n_features=None, n_splits=None):
        """
            Helper function to find the split with most information gain, by using find_rules, and information gain.

            data is a matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.

            label contains the corresponding labels. There might be multiple (i.e., > 2) classes.

            TODO: 9. Use find_rules to initialize rules variable
                  10. Initialize max_info to some negative number.
        """
        rules = self.find_rules(data)
        max_info = -float('inf')
        num_col = None
        threshold = None

        """
            TODO: 11. Check Number of features to use, None means all featurs. (Decision Tree always use all feature)
                      If n_features is a int, use n_features of features by random choice.
                      If n_features == 'sqrt', use sqrt(Total Number of Features ) by random choice.


        """
        if self.n_features is None:
            index_col = range(len(rules))
        elif isinstance(self.n_features, int):
            index_col = np.random.choice(
                len(rules), self.n_features, replace=False)
        else:
            num_features = int(np.sqrt(len(rules)))
            index_col = np.random.choice(
                len(rules), num_features, replace=False)

        """

            TODO: 12. Do the similar selection we did for features, n_split take in None or int or 'sqrt'.
                  13. For all selected feature and corresponding rules, we check it's information gain.

        """
        for col in index_col:
            for potential_threshold in rules[col]:
                info = self.information_gain(
                    label, data[:, col], potential_threshold)
                if info > max_info:
                    max_info = info
                    num_col = col
                    threshold = potential_threshold

        return threshold, num_col

    def build_tree(self, X, y, depth):
        """
                Helper function for building the tree.

                TODO: 14. First build the root node.
            """

        first_threshold, first_feature = self.next_split(X, y)
        current = self.Node(first_threshold, first_feature)

        """
                  TODO: 15. Base Case 1: Check if we pass the max_depth, \
                          check if the first_feature is None, min split size.
                            If some of those condition met, \
                                    change current to pure, \
                                    and set predict to the most popular label
                            and return current


              """

        if depth >= self.max_depth or \
                first_feature is None or len(X) <= self.size_allowed:

            current.predict = np.argmax(np.bincount(y.astype(int)))
            current.pure = True
            return current

            """
               Base Case 2: Check if there is only 1 label in this node, change current to pure, and set predict to the label
            """

        if len(np.unique(y)) == 1:
            current.predict = y[0]
            current.pure = True
            return current

            """
                TODO: 16. Find the left node index with feature i <= threshold  Right with feature i > threshold.
            """

        left_index = X[:, first_feature] <= first_threshold
        right_index = X[:, first_feature] > first_threshold

        """
                TODO: 17. Base Case 3: If we either side is empty, change current to pure, and set predict to the label
            """
        if not np.any(left_index) or not np.any(right_index):
            current.predict = np.argmax(np.bincount(y.astype(int)))
            current.pure = True
            return current

        left_X, left_y = X[left_index], y[left_index]
        current.left = self.build_tree(left_X, left_y, depth + 1)

        right_X, right_y = X[right_index], y[right_index]
        current.right = self.build_tree(right_X, right_y, depth + 1)

        return current

    def fit(self, X, y):
        """
            The fit function fits the Decision Tree model based on the training data. 

            X_train is a matrix or 2-D numpy array, represnting training instances. 
            Each training instance is a feature vector. 

            y_train contains the corresponding labels. There might be multiple (i.e., > 2) classes.
        """
        self.root = self.build_tree(X, y, 1)

        self.for_running = y[0]
        return self

    def ind_predict(self, inp):
        """
            Predict the most likely class label of one test instance based on its feature vector x.

            TODO: 18. Modify the following while loop to get the prediction.
                      Stop condition we are at a node is pure.
                      Trace with the threshold and feature.
                19. Change return self.for_running to appropiate value.
        """
        cur = self.root
        while not cur.pure:
            if inp[cur.feature] <= cur.threshold:
                cur = cur.left
            else:
                cur = cur.right
        return cur.predict

    def predict(self, inp):
        """
            X is a matrix or 2-D numpy array, represnting testing instances. 
            Each testing instance is a feature vector. 

            Return the predictions of all instances in a list.

            TODO: 20. Revise the following for-loop to call ind_predict to get predictions.
        """

        result = []
        for i in range(inp.shape[0]):
            result.append(self.ind_predict(inp[i]))
        return result
