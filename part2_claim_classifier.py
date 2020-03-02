import numpy as np
import torch as T
import pickle
import torch.optim as optim
import os
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import pandas as pd


class ClaimClassifier(T.nn.Module):

    def __init__(self, variables=9, multiplier=6, linear = False):
        super(ClaimClassifier, self).__init__()
        self.linear = linear
        if self.linear:
            self.linear = T.nn.Linear(variables, 1)
        else:
            self.multiplier = multiplier
            self.hid1 = T.nn.Linear(variables, multiplier * variables)  # 9-(8-8)-1
            self.hid2 = T.nn.Linear(multiplier * variables, variables)
            self.output = T.nn.Linear(variables, 1)
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        pass

    def forward(self, x):
        if self.linear:
            z = T.sigmoid(self.linear(x))
        else:
            z = T.tanh(self.hid1(x))
            z = T.tanh(self.hid2(z))
            z = T.sigmoid(self.output(z))
        return z

    def _preprocessor(self, X_raw, training=False):
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
        # YOUR CODE HERE
        X_raw=X_raw.to_numpy()

        if training:
            self.min = np.min(X_raw, axis=0)
            self.max = np.max(X_raw, axis=0)

        X_normed = (X_raw - self.min) / (self.max-self.min)

        return X_normed  # YOUR CLEAN DATA AS A NUMPY ARRAY

    def fit(self, new_train_x, new_train_y, loss, optimizer, max_epochs, n_batches):
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
        self.train()  # set training mode

        total_batches = round(len(new_train_x) / n_batches)

        for epoch in range(max_epochs):
            for i in range(total_batches):
                #   for data in batcher:
                #
                local_X, local_y = new_train_x[i * n_batches:(i + 1) * n_batches, ],\
                                   new_train_y[i * n_batches:(i + 1) * n_batches, ]
                # RANDOMLY SHUFFLE AND THEN SPLIT
                X = T.Tensor(local_X)
                Y = T.Tensor(local_y)

                # changed from optimizer to net.zero_grad
                self.zero_grad()

                output = self(X)

                loss_obj = loss(output, Y)

                loss_obj.backward()
                optimizer.step()

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
        X_clean = self._preprocessor(X_raw)
        X = T.Tensor(X_clean)
        oupt = self(X)
        pred_y = oupt >= 0.5529
        pred_y = pred_y.numpy()
        prob_y = oupt.detach().numpy()

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)

        # YOUR CODE HERE

        return pred_y  # YOUR PREDICTED CLASS LABELS

    def evaluate_architecture(self, data_x, data_y):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        # data_x and data_y are numpy array-of-arrays matrices

        data_x = pd.DataFrame(data=data_x)
        pred_y, prob_y = self.predict_probabilities(data_x)
        Y = T.ByteTensor(data_y)
        pred_y = T.from_numpy(pred_y)

        #num_correct = T.sum(Y == pred_y)
        #acc = (num_correct.item() * 100.0 / len(data_y))  # scalar
        acc = accuracy_score(data_y,pred_y)
        print('Accuracy: %f' % acc)
        con_matrix = confusion_matrix(data_y, pred_y)
        print(con_matrix)
        roc = roc_auc_score(data_y, prob_y)
        print('ROC: %f' % roc)
        return roc

    def save_model(self, model):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(model, target)

    def predict_probabilities(self, X_raw, pricing = False):
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
        if not pricing:
            X_clean = self._preprocessor(X_raw)
        else:
            X_clean = X_raw

        X = T.Tensor(X_clean)
        oupt = self(X)
        pred_y = oupt >= 0.5
        pred_y = pred_y.numpy()
        prob_y = oupt.detach().numpy()

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)

        # YOUR CODE HERE

        return pred_y, prob_y  # YOUR PREDICTED CLASS LABELS


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model


# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch(data_x, data_y, test_x, test_y, variables=9,pricing=False):
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class.

    The function should return your optimised hyper-parameters.
    """
    max_metric = 0
    searches = 100

    for i in range(searches):
        multiplier = round(np.random.uniform(10, 20))
        new_net = ClaimClassifier(variables=variables, multiplier=multiplier)
        lrn_rate = np.random.uniform(0.01, 0.2)

        loss = T.nn.BCELoss()
        no_batches = len(data_x)

        epochs = round(np.random.uniform(50, 150))
        new_net.train()
        new_net.min = np.min(data_x, axis=0)
        new_net.max = np.max(data_x, axis=0)
        optimizer = T.optim.Adam(new_net.parameters(), lr=lrn_rate)
        new_net.fit(data_x, data_y, loss, optimizer, epochs, no_batches)

        new_net.eval()
        print("Model (" + str(i + 1) + ") out of " + str(searches))
        metric = new_net.evaluate_architecture(test_x, test_y)

        if metric > max_metric:
            max_metric = metric
            best_lr = lrn_rate
            max_epochs = epochs
            best_multiplier = multiplier
            best_net = new_net
            if not pricing:
                new_net.save_model(new_net)

    return best_lr, max_epochs, best_multiplier, best_net


def create_heatmap(data):
    import plotly.graph_objects as go

    data = np.array(data)
    x = np.unique(data[:,0])
    y = np.unique(data[:,1])
    z = data[:,-1]
    z = np.reshape(z, (len(x), len(y)))
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y))
    fig.show()


def heatmap_data(data_x, data_y, test_x, test_y, variables = 9):
    data_for_graph = []


    multiplier = 1
    lrn_rate = 0.001
    # constants
    epochs = 50
    searches = 20
    for i in range(searches):
        for j in range(searches):

            new_net = ClaimClassifier(variables=variables, multiplier=multiplier)
            new_net.min = np.min(data_x, axis=0)
            new_net.max = np.max(data_x, axis=0)
            loss = T.nn.BCELoss()
            no_batches = len(data_x)

            new_net.train()
            optimizer = T.optim.Adam(new_net.parameters(), lr=lrn_rate)
            new_net.fit(data_x, data_y, loss, optimizer, epochs, no_batches)

            new_net.eval()
            metric = new_net.evaluate_architecture(test_x, test_y)
            data_for_graph.append([lrn_rate, multiplier, metric])

            # increment for the map
            multiplier += 1
        # increment for the map
        multiplier = 1
        lrn_rate += 0.005

    return (data_for_graph)



if __name__ == "__main__":

    # 1. load data

    # LOAD FULL FILE
    file = np.loadtxt("part2_training_data.csv", delimiter=',', skiprows=1, dtype=np.float32, ndmin=2)
    np.random.shuffle(file)

    x_data = file[:, :9]
    x_data = pd.DataFrame(data=x_data)
    net = ClaimClassifier()

    x_data = net._preprocessor(x_data,True)

    # Splitting data into 70% training, 15% validation, 15% test
    train_ratio = round(len(x_data) * 0.7)
    train_x, test_x = x_data[:train_ratio, :], x_data[train_ratio:, :]
    train_y, test_y = file[:train_ratio, 10:], file[train_ratio:, 10:]
    test_ratio = round(len(test_x) * 0.5)
    test_x, val_x = test_x[:test_ratio, :], test_x[test_ratio:, :]
    test_y, val_y = test_y[:test_ratio, :], test_y[test_ratio:, :]

    all_train_data = np.append(train_x, train_y, 1)
    (unique, counts) = np.unique(train_y, return_counts=True)
    np.random.shuffle(all_train_data)
    counter = 0

    #create_heatmap(heatmap_data(train_x, train_y, val_x, val_y))

    # UPSAMPLING
    z = np.copy(all_train_data)
    while (counts[0] > counts[1]):
        for i in all_train_data:
            if (counts[1] == counts[0]):
                break
            if (all_train_data[counter][-1] == unique[1]):
                z = np.vstack([z, all_train_data[counter][:]])
                counts[1] = counts[1] + 1
            counter = counter + 1
        counter = 0
    np.random.shuffle(z)
    new_train_x = z[:, :9]
    new_train_y = z[:, 9:]

    # Hyperparameter search
    best_lr, best_epochs, multiplier, best_net = \
        ClaimClassifierHyperParameterSearch(new_train_x, new_train_y, val_x, val_y)

    print("Best learning rate is: " + str(best_lr))
    print("Best multiplier is " + str(multiplier))
    print("Best epoch number is " + str(best_epochs))

    # Testing
    # net.fit(new_train_x, new_train_y, best_lr, best_loss_function, best_optimizer, best_epochs, best_no_batches)
    #    net.fit(new_train_x, new_train_y)

    # 4. evaluate model
    net = net.eval()  # set eval mode

    # # # # # # # # # # TESTING ON UNSEEN DATA # # # # # # # # # # # # # # # #
    trained_model = load_model()
    trained_model.eval()
    auc = trained_model.evaluate_architecture(test_x, test_y)
    print("AUC on test data = %0.2f%%" % auc)


    best_net.eval()
    auc = best_net.evaluate_architecture(test_x, test_y)
    print("AUC on test data = %0.2f%%" % auc)
    # 5. save model

