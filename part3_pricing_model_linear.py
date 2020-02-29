from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from part2_claim_classifier import ClaimClassifier
import torch
import torch.nn as nn
from torch import optim
import pandas as pd
import pickle
import numpy as np



def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


# class for part 3
class PricingModelLinear():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        self.base_classifier = ClaimClassifier()

    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
    def _preprocessor(self, X_raw, training = False):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE

        # Load simple data set used in part 2
        part2_headers = ["drv_age1", 'vh_age', 'vh_cyl', 'vh_din', 'pol_bonus', 'vh_sale_begin', 'vh_sale_end',
                         'vh_value', 'vh_speed', 'drv_age_lic1', 'pol_duration', 'pol_sit_duration', 'drv_age2']
        #  added from before
        # 'drv_age_lic1'
        #  pol_duration
        #  pol_sit_duration
        #  drv_age2

        required_attributes = X_raw[part2_headers]

        required_attributes = np.array(required_attributes)

        if training:
            self.means = np.mean(required_attributes, axis=0)
            self.std_dev = np.std(required_attributes, axis=0)

        x_normed = (required_attributes - self.means) / self.std_dev

        # Add extra columns here
        multiple_binarizers = []
        binarizer = LabelBinarizer()

        headers = ['drv_sex1', 'vh_type', 'pol_coverage', 'pol_usage']
        i = 0
        for header in headers:
            data = X_raw[header]
            if training:
                binarized = binarizer.fit_transform(data)
                multiple_binarizers.append(binarizer)
            else:
                binarized = self.saved_binarizers[i].transform(data)
            if len(binarized[0]) > 1:
                binarized = binarized[:, :-1]
            i += 1
            binarized = np.asarray(binarized)
            total = np.append(x_normed, binarized, axis=1)

        if training:
            self.saved_binarizers = multiple_binarizers

        return total

    def fit(self, X_raw, y_raw, claims_raw):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """
        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz])
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw, training=True)

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, X_clean, y_raw)
        else:
            self.base_classifier = find_best_model(X_clean,y_raw)

        return self.base_classifier

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)
        pred, prob_y = self.base_classifier.predict_probabilities(X_clean, pricing=True)

        return prob_y

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor

        premium_factor = 0.27
        premiums = self.predict_claim_probability(X_raw) * self.y_mean * premium_factor
        premiums = np.array(premiums)
        premiums = premiums.flatten()

        return premiums

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model_linear.pickle', 'wb') as target:
            pickle.dump(self, target)

def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part3_pricing_model_linear.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model


class LinearClaimClassifier(nn.Module):

    def __init__(self, variables=9):
        super(LinearClaimClassifier, self).__init__()
        self.linear = nn.Linear(variables,   1)
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        pass

    def forward(self, x):
        output = torch.sigmoid(self.linear(x))
        return output

    def predict(self, test_x):
        X = torch.Tensor(test_x)
        output = self(X)
        prob_y = output.detach().numpy()

        return prob_y

def find_best_model(x_clean,y_raw, variables=9, pricing=False):

    train_x, valid_x, train_y, valid_y = train_test_split(x_clean, y_raw, test_size=0.2)

    # Up sample
    (unique, counts) = np.unique(train_y, return_counts=True)
    total_train = np.append(train_x, train_y, axis=1)
    total_train = pd.DataFrame(total_train)

    df_class_0 = total_train[total_train.iloc[:, -1] == 0]
    df_class_1 = total_train[total_train.iloc[:, -1] == 1]

    total_train_class_1_over = df_class_1.sample(counts[0], replace=True)
    test_over = pd.concat([df_class_0, total_train_class_1_over], axis=0)

    total_train = np.array(test_over)

    new_train_y = total_train[:, -1]
    new_train_x = total_train[:, :-1]
    new_train_y = np.expand_dims(new_train_y, 1)
    max_metric = 0
    searches = 10

    for i in range(searches):
        new_net = ClaimClassifier(variables=len(train_x[0]),linear=True)
        lrn_rate = np.random.uniform(0.0001, 1)
        loss = nn.BCELoss()
        epochs = round(np.random.uniform(50, 150))
        new_net.train()
        optimizer = optim.SGD(new_net.parameters(), lr=lrn_rate)
        for j in range(epochs):
            X = torch.Tensor(new_train_x)
            Y = torch.Tensor(new_train_y)

            # changed from optimizer to net.zero_grad
            new_net.zero_grad()

            output = new_net(X)

            loss_obj = loss(output, Y)

            loss_obj.backward()
            optimizer.step()

        new_net.eval()
        print("Model (" + str(i + 1) + ") out of " + str(searches))
        pred, probabilities = new_net.predict_probabilities(valid_x,pricing=True)
        metric = roc_auc_score(valid_y, probabilities)
        print("Roc Score:" + str(metric))
        if metric > max_metric:
            max_metric = metric
            best_lr = lrn_rate
            max_epochs = epochs
            best_net = new_net

    return best_net

if __name__ == "__main__":

    dat = pd.read_csv("part3_training_data.csv")
    attributes = dat.drop(columns=["claim_amount", "made_claim"])
    y = dat["made_claim"]
    claim_amounts = dat['claim_amount']
    X, Y = attributes, y

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.15)
    train_x = pd.DataFrame(train_x)
    train_y = pd.DataFrame(train_y)

    linear_model = PricingModelLinear()

    linear_model.fit(train_x, train_y, claim_amounts)
    probs = linear_model.predict_claim_probability(test_x)

    linear_model_roc = roc_auc_score(test_y,probs)

    print("Final Model Roc Score: " + str(linear_model_roc))
    print("")
    print("Saving...")
    linear_model.save_model()
    print("Loading...")
    loaded_model = load_model()
    loaded_probs = loaded_model.predict_claim_probability(test_x)
    loaded_model_roc = roc_auc_score(test_y,loaded_probs)
    print("Loaded Model Roc Score: " + str(linear_model_roc))
    loaded_model_prices = loaded_model.predict_premium(test_x)
    print(loaded_model_prices)