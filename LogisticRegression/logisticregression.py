import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd


def min_max_normalize(lst):
    """
        Helper function for movielens dataset, not useful for discrete multi class clasification.

        Return:
        Normalized list x, in range [0, 1]
    """
    maximum = max(lst)
    minimum = min(lst)
    toreturn = []
    for i in range(len(lst)):
        toreturn.append((lst[i] - minimum) / (maximum - minimum))
    return toreturn


def z_standardize(X_inp):
    """
        Z-score Standardization.
        Standardize the feature matrix, and store the standarize rule.

        Parameter:
        X_inp: Input feature matrix.

        Return:
        Standardized feature matrix.
    """

    toreturn = X_inp.copy()
    for i in range(X_inp.shape[1]):
        # ------ Find the standard deviation of the feature
        std = np.std(X_inp[:, i])
        # ------ Find the mean value of the feature
        mean = np.mean(X_inp[:, i])
        """
        #TODO: 1. implement the standardize function
        """
        if std == 0:
            # Avoid division by zero; if std is 0, the feature is constant and can be set to 0 directly
            temp = np.zeros_like(X_inp[:, i])
        else:
            temp = (X_inp[:, i] - mean) / std  # Standardize the feature
        toreturn[:, i] = temp
    return toreturn


def sigmoid(x):
    """
        Sigmoid Function

        Return:
        transformed x.
    """
    """
        #TODO: 2. implement the sigmoid function
    """
    return 1 / (1 + np.exp(-x))


class Logistic_Regression():

    def __init__(self, early_stop=0, standardized=True):
        """
            Some initializations, if neccesary
        """
        self.early_stop = early_stop
        self.standardized = standardized
        self.theta, self.b = 0, 0
        self.X, self.y = None, None
        self.model_name = 'Logistic Regression'

    def fit(self, X_train, y_train):
        """
            Save the datasets in our model, and do normalization to y_train

            Parameter:
                X_train: Matrix or 2-D array. Input feature matrix.
                Y_train: Matrix or 2-D array. Input target value.
        """

        self.X = X_train
        self.y = y_train

        count = 0
        uni = np.unique(y_train)
        for y in y_train:
            if y == min(uni):
                self.y[count] = -1
            else:
                self.y[count] = 1
            count += 1

        n, m = X_train.shape
        self.theta = np.zeros(m)
        self.b = 0

    def gradient(self, X_inp, y_inp, theta, b):
        """
            Calculate the grandient of Weight and Bias, given sigmoid_yhat, true label, and data

            Parameter:
                X_inp: Matrix or 2-D array. Input feature matrix.
                y_inp: Matrix or 2-D array. Input target value.
                theta: Matrix or 1-D array. Weight matrix.
                b: int. Bias.

            Return:
                grad_theta: gradient with respect to theta
                grad_b: gradient with respect to b

        NOTE: There are several ways of implementing the gradient. We are merely providing you one way
        of doing it. Feel free to change the code and implement the way you want.
        """
        grad_b = 0
        grad_theta = np.zeros(len(theta))

        """
            TODO: 3. Update grad_b and grad_theta using the Sigmoid function
        """
        m = X_inp.shape[0]  # Number of examples
        z = np.dot(X_inp, theta) + b
        y_hat = sigmoid(z)  # Predicted probabilities
        error = y_hat - y_inp  # Difference between predicted and actual values
        grad_theta = np.dot(X_inp.T, error) / m
        grad_b = np.sum(error) / m

        return grad_theta, grad_b

    def gradient_descent_logistic(self, alpha, num_pass, early_stop=0, standardized=True):
        """
            Logistic Regression with gradient descent method

            Parameter:
                alpha: (Hyper Parameter) Learning rate.
                num_pass: Number of iteration
                early_stop: (Hyper Parameter) Least improvement error allowed before stop.
                            If improvement is less than the given value, then terminate the function and store the coefficents.
                            default = 0.
                standardized: bool, determine if we standardize the feature matrix.

            Return:
                self.theta: theta after training
                self.b: b after training
        """

        if standardized:
            self.X = z_standardize(self.X)

        # n, m = self.X.shape

        prev_loss = float('inf')

        for i in range(num_pass):

            """
                TODO: 4. Modify the following code to implement gradient descent algorithm
            """
            grad_theta, grad_b = self.gradient(
                self.X, self.y, self.theta, self.b)
            self.theta -= alpha * grad_theta
            self.b -= alpha * grad_b
            """
                TODO: 5. Modify the following code to implement early Stop Mechanism (use Logistic Loss when calculating error)
            """
            # Implement logistic loss to compute error and apply early stopping
            y_hat = sigmoid(np.dot(self.X, self.theta) + self.b)
            curr_loss = -np.mean(self.y * np.log(y_hat) +
                                 (1 - self.y) * np.log(1 - y_hat))
            if np.abs(prev_loss - curr_loss) < early_stop:
                break
            prev_loss = curr_loss

        return self.theta, self.b  # Return final coefficients after all iterations

    def predict_ind(self, x: list):
        """
            Predict the most likely class label of one test instance based on its feature vector x.

            Parameter:
            x: Matrix, array or list. Input feature point.

            Return:
                p: prediction of given data point
        """

        """
            TODO: 7. Implement the prediction function
        """

        # Calculate probability
        p = sigmoid(np.dot(x, self.theta) + self.b)

        return p

    def predict(self, X):
        """
            X is a matrix or 2-D numpy array, represnting testing instances.
            Each testing instance is a feature vector.

            Parameter:
            x: Matrix, array or list. Input feature point.

            Return:
                p: prediction of given data matrix
        """

        """
            TODO: 8. Revise the following for-loop to call predict_ind to get predictions.
        """

        if self.standardized:
            X = np.array([z_standardize(x) for x in X.T]).T

        # Use predict_ind to generate the prediction list
        return [self.predict_ind(x) for x in X]
