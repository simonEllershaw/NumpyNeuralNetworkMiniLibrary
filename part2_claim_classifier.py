import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random


class ClaimClassifier():

    def __init__(self):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        pass

    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """

        # Calculate mean and standard dev for each column
        means = np.mean(X_raw, axis=0)
        std_dev = np.std(X_raw, axis=0)

        # Apply Normalisation to the data
        for i in range(len(X_raw)):
            for j in range(len(X_raw[0])):
                X_raw[i, j] = X_raw[i, j] - means[j]
                X_raw[i, j] = X_raw[i, j]/std_dev[j]

        # Apply cross validation?
        """
        k=10 # Splits for data
        X_raw_splits = np.split(X_raw,k)
        for i in range(8)
            np.stack(.....)
         
        validation_set = X_raw_splits[8]
        testing_set = X_raw_splits[9]
        """
        return X_raw

    def fit(self, X_raw, y_raw):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)
        # YOUR CODE HERE
        pass

    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)

        # YOUR CODE HERE

        return  # YOUR PREDICTED CLASS LABELS

    def evaluate_architecture(self):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        pass

    def save_model(self):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model


# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch(training_set, testing_set):
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """
    best_lr = 0
    best_momentum = 0
    max_metric = 0

    for i in range(60):
        lr = random.uniform(0, 1)
        momentum = random.uniform(0.5, 1)
        loss = random.uniform(0, 1)
        if round(loss) == 1:
            loss_function = nn.BCELoss()
        else:
            loss_function = nn.HingeEmbeddingLoss()

        optimiser = optim.SGD(lnet.parameters(),lr=lr, momentum=momentum)

        metric = ...
        if metric > max_metric:
            best_lr = lr
            best_momentum = momentum

    return best_lr, best_momentum


if __name__ == "__main__":
    # Open csv file
    raw_data = np.genfromtxt("./part2_training_data.csv", delimiter=",")
    attributes = raw_data[1:, :9]
    labels = raw_data[1:, 10:]

    # Instantiate classifier and pre-process data
    myClassifier = ClaimClassifier()
    X_raw = myClassifier._preprocessor(attributes)

    labels_summary = np.unique(labels, return_counts=True)

    print(labels_summary)


