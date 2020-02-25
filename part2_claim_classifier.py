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

    def __init__(self):
        super(ClaimClassifier, self).__init__()
        self.hid1 = T.nn.Linear(9, 12)  # 9-(8-8)-1
        self.hid2 = T.nn.Linear(12, 12)
        self.oupt = T.nn.Linear(12, 1)
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        pass

    def forward(self, x):
        z = T.tanh(self.hid1(x))
        z = T.tanh(self.hid2(z))
        z = T.sigmoid(self.oupt(z))
        return z

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
        # YOUR CODE HERE
        min_max_scaler = preprocessing.MinMaxScaler()
        X_normed = min_max_scaler.fit_transform(X_raw)

        return X_normed # YOUR CLEAN DATA AS A NUMPY ARRAY


    def fit(self, new_train_x, new_train_y,lrn_rate,loss,optimizer,max_epochs,n_batches):
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
        """
        lrn_rate = 0.001
        loss = T.nn.BCELoss()  # softmax() + binary CE
        #     loss_func=T.nn.HingeEmbeddingLoss()
        # loss = T.nn.MSELoss()
        # Alternative optimizwer method
        # optimizer=T.optim.Adam(net.parameters(),lr=lrn_rate)
        # microsoft optimiser method
        # SGD= stochastic gradient descent
        optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
        max_epochs = 3
        n_items = len(new_train_x)
        n_batches = 10
        """
        total_batches = round(len(new_train_x) / n_batches)


        for epoch in range(max_epochs):
            for i in range(total_batches):
                #   for data in batcher:
                #
                local_X, local_y = new_train_x[i * n_batches:(i + 1) * n_batches, ], new_train_y[
                                                                                     i * n_batches:(i + 1) * n_batches, ]
                # RANDOMLY SHUFFLE AND THEN SPLIT
                X = T.Tensor(local_X)
                Y = T.Tensor(local_y)

                # changed from optimizer to net.zero_grad
                self.zero_grad()
                oupt = self(X)
                #  print(oupt)
                # changing loss function to MSE
                loss_obj = loss(oupt, Y)
                # print(loss_obj)
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

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)

        # YOUR CODE HERE

        return  # YOUR PREDICTED CLASS LABELS

    def evaluate_architecture(self,data_x, data_y):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        # data_x and data_y are numpy array-of-arrays matrices
        X = T.Tensor(data_x)

        Y = T.ByteTensor(data_y)  # a Tensor of 0s and 1s
        oupt = self(X)  # a Tensor of floats
        pred_y = oupt >= 0.5  # a Tensor of 0s and 1s
        num_correct = T.sum(Y == pred_y)  # a Tensor
        acc = (num_correct.item() * 100.0 / len(data_y))  # scalar
        print('Accuracy: %f' % acc)
        con_matrix = confusion_matrix(data_y, pred_y)
        # precision tp / (tp + fp)
        precision = precision_score(data_y, pred_y)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(data_y, pred_y)
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(data_y, pred_y)
        print('F1 score: %f' % f1)
        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(data_y, pred_y)
        print('Accuracy: %f' % accuracy)
        print(con_matrix)
        roc = roc_auc_score(data_y, pred_y)
        print('ROC: %f' % roc)
        return (roc)



    def save_model(self,model):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(model, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model


# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch(data_x, data_y,test_x,test_y):
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class.

    The function should return your optimised hyper-parameters.
    """
    max_metric=0
    max_loss= 100000
    for i in range(60):
        new_net = ClaimClassifier()
        lrn_rate = np.random.uniform(0, 0.1)
        momentum = np.random.uniform(0.5, 1)
        loss_no = np.random.uniform(0, 1)
        if round(loss_no) == 1:
            loss = T.nn.BCELoss()
        else:
            loss = T.nn.HingeEmbeddingLoss()
        no_batches = round(np.random.uniform(10, 100))
        epochs = round(np.random.uniform(10, 100))

        optimizer = T.optim.SGD(new_net.parameters(), lr=lrn_rate, momentum=momentum)
        new_net.fit(data_x, data_y,lrn_rate,loss,optimizer,epochs,no_batches)
        metric = new_net.evaluate_architecture(test_x, test_y)

        if metric > max_metric:
            best_lr = lrn_rate
            best_momentum = momentum
            best_loss_function = loss
            max_metric = metric
            max_epochs = epochs
            max_no_batches = no_batches
            new_net.save_model(new_net)




    return best_lr, best_momentum, best_loss_function, optimizer, max_epochs, max_no_batches



if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # 0. get started
    #  T.manual_seed(1)
    #np.random.seed(1)
    # 1. load data

    # LOAD FULL FILE
    file = np.loadtxt("part2_training_data.csv", delimiter=',', skiprows=1, dtype=np.float32, ndmin=2)
    np.random.shuffle(file)
    x_data = file[:, :9]
    net=ClaimClassifier()
    x_data=net._preprocessor(x_data)

    #Splitting data into 70% training, 15% validation, 15% test
    train_ratio=round(len(x_data)*0.7)
    train_x, test_x = x_data[:train_ratio,:], x_data[train_ratio:,:]
    train_y, test_y = file[:train_ratio, 10:], file[train_ratio:, 10:]
    test_ratio=round(len(test_x)*0.5)
    test_x, val_x = test_x[:test_ratio, :], test_x[test_ratio:, :]
    test_y, val_y = test_y[:test_ratio, :], test_y[test_ratio:, :]

    all_train_data = np.append(train_x, train_y, 1)
    (unique, counts) = np.unique(train_y, return_counts=True)
    np.random.shuffle(all_train_data)
    counter = 0

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

    #Hyperparameter search
    best_lr, best_momentum, best_loss_function,best_optimizer, best_epochs, best_no_batches=\
        ClaimClassifierHyperParameterSearch(new_train_x,new_train_y,val_x, val_y)

    #Testing
    net.fit(new_train_x,new_train_y,best_lr,best_loss_function,best_optimizer,best_epochs,best_no_batches)
    #    net.fit(new_train_x, new_train_y)

    # 4. evaluate model
    net = net.eval()  # set eval mode
    acc=net.evaluate_architecture(test_x, test_y)
    print("Accuracy on test data = %0.2f%%" % acc)
    # 5. save model






