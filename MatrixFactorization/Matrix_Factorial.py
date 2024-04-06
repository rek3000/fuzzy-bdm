import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Matrix_Factorization:
    def __init__(self, alpha=0.00001, iterations=50, num_of_latent=200, lam=0.0005):
        """
        Some initializations, if neccesary

        attributes:
                    alpha: Learning Rate, default 0.01
                    num_iter: Number of Iterations to update coefficient with training data
                    num_of_latent: Number of latent factor.
                    lam: Regularization constant


        TODO: 1. Initialize all variables needed.
        """

        self.alpha = alpha
        self.iterations = iterations
        self.num_of_latent = num_of_latent
        self.lam = lam

    def fit(self, train):
        """
        Train: list of tuples with (User, Movie, Rating)
        num_user: Number of unique user.
        num_movie: Number of unique movie

        TODO: 2. Initialize num_user and num_movie
              3. Save the training set.
              4. Initialize P and Q matrix, with normal distribution with mean = 0.
              Hint: Think about what P and Q represent, what they should do.Think about the shape too.


        """
        num_user = len(set([t[0] for t in train]))
        num_movie = len(set([t[1] for t in train]))

        self.train = train

        self.P = np.random.normal(
            0, 0.1, (num_user + 1, self.num_of_latent)
        )  # plus 1 to use index from 1
        self.Q = np.random.normal(0, 0.1, (num_movie + 1, self.num_of_latent))

        rmse_lst = []

        """
            TODO: 5: Calculate the error, using P and Q matrix.
                  6: We need to check if the absolute value error is less than some constant. Store the previous Q and P for adaptive learning rate.
                      If it is less than that constant then we update P and Q matrix.
                      (When update, update the P and Q at the same time. Think about why it is important.)
                      Otherwise use the error to update the Q and P matrix.

                  7: For each entry update temp_mse, and append the Current iteration RMSE to rmse_lst.

        """

        for f in range(self.iterations):
            ### Random Shuffle. Why is this called?
            np.random.shuffle(self.train)

            temp_mse = 0

            previous_Q = self.Q.copy()
            previous_P = self.P.copy()

            Count = 0

            for tup in self.train:
                u, i, rating = tup
                error = rating - np.dot(self.P[u], self.Q[i])

                if abs(error) > 20:
                    continue
                Count += 1
                temp_mse += error**2 + self.lam * (
                    np.linalg.norm(self.P[u]) ** 2 + np.linalg.norm(self.Q[i]) ** 2
                )

                P = self.P[u] + self.alpha * (error * self.Q[i] - self.lam * self.P[u])
                Q = self.Q[i] + self.alpha * (error * self.P[u] - self.lam * self.Q[i])

                #### Don't Modify this code, helpful for converge.
                if (
                    np.isinf(Q).any()
                    or np.isinf(P).any()
                    or np.isnan(Q).any()
                    or np.isnan(P).any()
                ):
                    pass
                else:
                    # pass
                    #### NEED TO MODIFY #### Update P and Q.
                    self.Q[i] = Q
                    self.P[u] = P

            rmse_lst.append(np.sqrt(temp_mse / Count))

            """
                TODO: 8: Implement the adaptive learning rate.
                         If current rmse is less than previous iteration, let's increase by a factor range from 1 - 1.5
                         Otherwise we decrease by a factor range from 0.5 - 1
                      9: If the current rmse is greater than previous iteration.
                         Check the relative error, (previous - current)/ previous.
                         If it is greater than 0.1, we restore the previous Q and P. (Try without it. Think about why we need this.)
            """
            if len(rmse_lst) <= 1:
                continue

            if rmse_lst[-1] < rmse_lst[-2]:
                self.alpha *= np.random.uniform(1, 1.5)
            else:
                self.alpha *= np.random.uniform(0.5, 1)
                if (rmse_lst[-2] - rmse_lst[-1]) / rmse_lst[-2] > 0.1:
                    self.Q = previous_Q
                    self.P = previous_P

        self.rmse = rmse_lst

    def ind_predict(self, tup):
        """
        tup: One single entry, (user, movie)

        TODO: 10: Use P and Q to make prediction on single entry.

        """
        u, i, _ = tup
        return np.dot(self.P[u], self.Q[i])

    def predict(self, X):
        """
        X: list of entries

        TODO: 11: Use ind_predict we create to make predicitons.
        """
        res = []
        for i in X:
            res.append(self.ind_predict(i))
        return res
