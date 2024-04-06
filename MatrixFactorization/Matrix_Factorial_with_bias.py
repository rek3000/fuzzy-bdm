import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Matrix_Factorization_with_bias:
    def __init__(self, alpha=0.00001, iterations=50, num_of_latent=200, lam=0.01):
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

        TODO: 2. Initialize num_user and num_movie.
              3. Save the training set.
              4. Initialize bu , bi and b. b is the global mean of the rating.
              5. Initialize P and Q matrix.
              Hint: Think about what P and Q represent, what they should do.Think about the shape too.



        """

        num_user = len(set([t[0] for t in train]))
        num_movie = len(set([t[1] for t in train]))
        self.train = train

        self.P = np.random.normal(0, 0.1, (num_user + 1, self.num_of_latent))
        self.Q = np.random.normal(0, 0.1, (num_movie + 1, self.num_of_latent))
        self.bu = np.zeros(num_user + 1)
        self.bi = np.zeros(num_movie + 1)
        self.b = np.mean([t[2] for t in train])

        rmse_lst = []

        """
            TODO: 5: Calculate the error, using P , Q , bu , bi and b.
                  6: Update the P , Q , bu , bi and b with error you calculate.
                    (Think about why we don't need to check the absolute of error)
                  7: For each entry update temp_mse, and append the Current iteration RMSE to rmse_lst.

        """

        for f in range(self.iterations):

            np.random.shuffle(self.train)

            temp_mse = 0
            previous_Q = self.Q.copy()
            previous_P = self.P.copy()
            previous_bu = self.bu.copy()
            previous_bi = self.bi.copy()

            Count = 0
            for tup in self.train:
                u, i, rating = tup

                prediction = (
                    self.b + self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])
                )
                error = rating - prediction
                temp_mse += error**2 + self.lam * (
                    np.linalg.norm(self.P[u]) ** 2
                    + np.linalg.norm(self.Q[i]) ** 2
                    + self.bu[u] ** 2
                    + self.bi[i] ** 2
                )
                Count += 1

                self.bu[u] += self.alpha * (error - self.lam * self.bu[u])
                self.bi[i] += self.alpha * (error - self.lam * self.bi[i])
                self.b += self.alpha * (error - self.lam * self.b)

                self.P[u] += self.alpha * (error * self.Q[i] - self.lam * self.P[u])
                self.Q[i] += self.alpha * (error * self.P[u] - self.lam * self.Q[i])

            rmse_lst.append(np.sqrt(temp_mse) / Count)

            """
                TODO: 8: Implement the adaptive learning rate.
                         If current rmse is less than previous iteration, let's increase by a factor range from 1 - 1.5
                         Otherwise we decrease by a factor range from 0.5 - 1
                      9: If the current rmse is greater than previous iteration.
                         Check the relative error, (previous - current)/ previous.
                         If it is greater than 0.1, we restore the previous Q and P. (Try without it. Think about why we need this.)
            """

            if len(rmse_lst) >= 2:
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
        return self.b + self.bu[u] + self.bi[i] + np.dot(self.P[u], self.Q[i])

    def predict(self, X):
        """
        X: list of entries

        TODO: 11: Use ind_predict we create to make predicitons.
        """
        res = []
        for i in X:
            res.append(self.ind_predict(i))
        return res
