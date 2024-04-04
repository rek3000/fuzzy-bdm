import numpy as np
import pandas as pd

class Naive_Bayes():
    """
    
    Naive Bayes classifer
    
    Attributes:
        prior: P(Y)
        likelihood: P(X_j | Y)
    """
    
    def __init__(self):
        """
            Some initializations, if neccesary
        """
        
        self.model_name = 'Naive Bayes'
        self.prior = {}
        self.likelihood = {}
    
    def fit(self, X_train, y_train):
        
        """ 
            The fit function fits the Naive Bayes model based on the training data. 
            Here, we assume that all the features are **discrete** features. 
            
            X_train is a matrix or 2-D numpy array, represnting training instances. 
            Each training instance is a feature vector. 

            y_train contains the corresponding labels. There might be multiple (i.e., > 2) classes.
        """
        
        """
            FINISHED_TODO: 1. Modify and add some codes to the following for-loop
                     to compute the correct prior distribution of all y labels.
                  2. Make sure they are normalized to a distribution.
        """
        """
            FINISHED_TODO: 3. Modify and add some codes to the following for-loops
                     to compute the correct likelihood P(X_j | Y).
                  4. Make sure they are normalized to distributions.
        """
        
        X_train = np.array(X_train)  # Convert X_train to numpy array
        self.compute_prior(X_train, y_train)
        self.compute_likelihood(X_train, y_train.astype(str))  # Convert y_train to strings

        """
            FINISHED_TODO: 5. Think about whether we really need P(X_1 = x_1, X_2 = x_2, ..., X_d = x_d)
                     in practice?
                  6. Does this really matter for the final classification results?
        """

        """
            ANSWER TODO-QUESTION: 5. Well, not really. It's like trying to track every possible combination of features. 
                Not only is it computationally expensive, but it also goes against the spirit of Naive Bayes, 
                which assumes that features are independent given the class label. 
                We can get away with just looking at the probability of each feature given the class.

            ANSWER TODO-QUESTION: 6. Surprisingly, not much! Naive Bayes might be naive, but it's still pretty effective, 
                especially in tasks like text classification. 
                Even though it simplifies by assuming feature independence, it often delivers decent results. 
                However, in situations where features are highly dependent on each other, 
                Naive Bayes might not shine as brightly compared to other models that can handle those dependencies better. 
                So, while it's a simplification, 
                it's good to keep in mind its limitations and consider your data's nature when choosing a classifier.
        """
        
    def compute_prior(self, X_train, y_train):
        class_counts = {}
        total_samples = len(y_train)
        
        for y in y_train:
            if y in class_counts:
                class_counts[y] += 1
            else:
                class_counts[y] = 1
        
        for y, count in class_counts.items():
            self.prior[y] = count / total_samples
    
    def compute_likelihood(self, X_train, y_train):
        feature_counts = {}
        class_counts = {}
        
        for x, y in zip(X_train, y_train):
            for j, value in enumerate(x):
                if (j, value, y) in feature_counts:
                    feature_counts[(j, value, y)] += 1
                else:
                    feature_counts[(j, value, y)] = 1

                if (j, y) in class_counts:
                    class_counts[(j, y)] += 1
                else:
                    class_counts[(j, y)] = 1
        
        for (j, value, y), count in feature_counts.items():
            self.likelihood[(j, value, y)] = count / class_counts[(j, y)]
      
    def ind_predict(self, x : list):
        
        """ 
            Predict the most likely class label of one test instance based on its feature vector x.
        """
        
        """
            FINISHED_TODO: 7. Enumerate all possible class labels and compute the likelihood 
                     based on the given feature vector x. Don't forget to incorporate 
                     both the prior and likelihood.
                  8. Pick the label with the higest probability. 
                  9. How to deal with very small probability values, especially
                     when the feature vector is of a high dimension. (Hint: log)
                  10. How to how to deal with unknown feature values?
        """
        
        """
            Dealing with unknown feature values is like dealing with surprises in your dataset you didn't expect them, 
            but you have to handle them gracefully. 
            When our model encounters an unknown feature value during prediction, we've got a few tricks up our sleeve. 
            One option is to simply ignore the unknown feature and move on with the prediction. 
            This works well if the unknown feature isn't a big deal in deciding the outcome. 
            Another approach is to play it safe and assign a small probability to the unknown value. 
            This way, it doesn't mess up our prediction too much, but we still acknowledge its existence. 
            Then there's smoothing it's like adding a sprinkle of seasoning to even out the flavor. 
            Techniques like Laplace smoothing add a tiny count to all possible feature values, including the unknowns, 
            to prevent us from getting stuck with zero probabilities. 
            And if we're feeling adventurous, we can try imputing the missing values based on what we know from the rest of the data. 
            It's all about finding the right balance between handling surprises and staying true to the data. 
            So, in the end, it's a bit of trial and error to see which approach works best for our model and the problem at hand.
        """
        
        max_prob = -1
        best_class = None
        
        for y, prior in self.prior.items():
            likelihood = 1
            for j, value in enumerate(x):
                if (j, value, y) in self.likelihood:
                    likelihood *= self.likelihood[(j, value, y)]
                else:
                    likelihood = 0  # handling unknown values
            probability = prior * likelihood
            if probability > max_prob:
                max_prob = probability
                best_class = y
        return best_class
    
    def predict(self, X):
        
        """
            X is a matrix or 2-D numpy array, represnting testing instances. 
            Each testing instance is a feature vector. 
            
            Return the predictions of all instances in a list.
        """
        
        """
            FINISHED_TODO: 11. Revise the following for-loop to call ind_predict to get predictions.
        """
        
        X = np.array(X)  # Convert X to numpy array
        predictions = []
        for x in X:
            predictions.append(self.ind_predict(x))
        return predictions
