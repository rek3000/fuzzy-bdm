class KMeans():
    def __init__(self, k = 3, num_iter = 1000):
        """
            Some initializations, if neccesary

            Parameter:
                k: Number of clusters we are trying to classify
                num_iter: Number of iterations we are going to loop
        """

        self.model_name = 'KMeans'
        self.k = k
        self.num_iter = num_iter
        self.centers = None
        self.RM = None

    def train(self, X):
        """
            Train the given dataset

            Parameter:
                X: Matrix or 2-D array. Input feature matrix.

            Return:
                self: the whole model containing relevant information
        """

        r, c = X.shape
        centers = []
        RM = np.zeros((r, self.k))

        """
            TODO: 1. Modify the following code to randomly choose the initial centers
        """
        initials = np.random.choice(r, self.k, replace=False)
        centers = X[initials]

        for i in range(self.num_iter):
            for j in range(r):
                """
                    TODO: 2. Modify the following code to update the Relation Matrix
                """
                distance = np.linalg.norm(X[j] - centers, axis=1)
                minpos = np.argmin(distance)

                temp_rm = np.zeros(self.k)
                temp_rm[minpos] = 1
                RM[j,:] = temp_rm
            new_centers = np.zeros_like(centers)
            for l in range(self.k):
                """
                    TODO: 3. Modify the following code to update the centers
                """
                row_index = (RM[:, l] == 1).flatten()
                all_l = X[row_index, :]
                if len(all_l) > 0:
                  new_centers[l, :] = np.mean(all_l, axis=0)
                else:
                    new_centers[l, :] = centers[l, :]
            if np.sum(new_centers - centers) < 0.0001:
                self.centers = new_centers
                self.RM = RM
                return self
            centers = new_centers
        self.centers = centers
        self.RM = RM
        return self
