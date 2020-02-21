import numpy as np
import pickle


def xavier_init(size, gain=1.0):
    """
    Xavier initialization of network weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative log-
    likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs
        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class ActivationLondon(Layer):
    """
    ActivationLayer: Abstract class, applies activation function elementwise
    """

    def __init__(self, *args, **kwargs):
        self._cache_current = None

    def activationFunction(self, x):
        """ Applies activation function elementwise to an array x """
        raise NotImplementedError()

    def derivativeOfActivationFunction(self, x):
        """ Applies derivative of activation function elementwise to an array"""
        raise NotImplementedError()

    def forward(self, x):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # storing partial derivative of x for backpropagation
        self._cache_current = self.derivativeOfActivationFunction(x)

        # returning forward pass
        return self.activationFunction(x)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        activationFunctionDerivatives = self._cache_current

        # Hadamard product used here
        return np.multiply(grad_z, activationFunctionDerivatives)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class SigmoidLayer(ActivationLondon):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def activationFunction(self, x):
        """ Applies sigmoid function elementwise to an array x """
        x = np.array(x)
        return np.array((1 / (1 + np.exp(np.negative(x)))))

    def derivativeOfActivationFunction(self, x):
        """ Applies derivative of sigmoid function elementwise to an array"""
        sigmoid = self.activationFunction(x)
        return sigmoid * (1 - sigmoid)


class ReluLayer(ActivationLondon):
    """
    ReluLayer: Applies Relu function elementwise.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def activationFunction(self, x):
        """ Applies ReluLayer function elementwise to an array x """
        return np.where(x > 0, x, 0.0)

    def derivativeOfActivationFunction(self, x):
        """ Applies derivative of ReluLayer function elementwise to an array"""
        return np.where(x > 0, 1.0, 0.0)


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """Constructor.

        Arguments:
            n_in {int} -- Number (or dimension) of inputs.
            n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # initialise the weights before training using xavier_init method
        self._W = xavier_init([self.n_in, self.n_out])

        # initialise the biases to zero before training
        self._b = np.zeros([self.n_out], dtype=float)

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        x = np.asarray(x)

        # storing input array for backpropagation
        self._cache_current = x

        # perform forward pass
        return np.matmul(x, self._W) + self._b
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        grad_z = np.array(grad_z)
        x = self._cache_current

        # set up column vector of ones matrix transposed
        batch_size = len(grad_z)
        ones = np.ones(batch_size)

        # calculate gradients of linear parameters
        # Note x values used in forward propogation are stored in _cache_current
        self._grad_W_current = np.matmul(x.T, grad_z)
        self._grad_b_current = np.matmul(ones, grad_z)

        # return gradient of loss for inputs
        return np.matmul(grad_z, self._W.T)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # perform weight's update
        self._W -= np.dot(self._grad_W_current, learning_rate)

        # perform biases' update
        self._b -= np.dot(self._grad_b_current, learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """Constructor.

        Arguments:
            input_dim {int} -- Dimension of input (excluding batch dimension).
            neurons {list} -- Number of neurons in each layer represented as a
                list (the length of the list determines the number of layers).
            activations {list} -- List of the activation function to use for
                each layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # argument error handling
        if len(neurons) != len(activations):
            raise ValueError("The length of the neuron and activations list must be equal")
        elif len(neurons) < 1:
            raise ValueError("Must have at least 1 layer the network")

        # initialise parameters
        self._layers = []
        n_in = input_dim

        # populate layers
        for index in range(len(neurons)):
            n_out = neurons[index]
            # add linear layer
            self._layers.append(LinearLayer(n_in, n_out))
            # add activation layer
            activationLayer = self.string_to_activation_Layer(activations[index])
            if activationLayer:
                self._layers.append(activationLayer)
            # set n_in for next iteration
            n_in = n_out
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    @staticmethod
    def string_to_activation_Layer(layer_name):
        """ Returns instance of an activation layer from a string of layers name
            Raises value error if invalid layer name given"""
        switcher = {
            "relu": ReluLayer(),
            "sigmoid": SigmoidLayer(),
            "identity": None
        }
        result = switcher.get(layer_name, "INVALID")
        if result == "INVALID":
            raise ValueError(layer_name + " is an invalid layer name")
        return result

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        forward_output = x

        # iterate through the layers, updating forward_output accordingly
        for layer in self._layers:
            forward_output = layer.forward(forward_output)

        return forward_output

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (1,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, input_dim).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        backward_output = grad_z

        # iterate backwards through the layers, updating backward_output accordingly
        for layer in reversed(self._layers):
            backward_output = layer.backward(backward_output)

        return backward_output

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        for layer in self._layers:
            layer.update_params(learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
            self,
            network,
            batch_size,
            nb_epoch,
            learning_rate,
            loss_fun,
            shuffle_flag,
    ):
        """Constructor.

        Arguments:
            network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            batch_size {int} -- Training batch size.
            nb_epoch {int} -- Number of training epochs.
            learning_rate {float} -- SGD learning rate to be used in training.
            loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._loss_layer = self.string_to_loss_layer(self.loss_fun)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    @staticmethod
    def string_to_loss_layer(loss_function_name):
        """ Returns instance of an loss function layer from a string of layers name
            Raises value error if invalid layer name given"""
        switcher = {
            "mse": MSELossLayer(),
            "cross_entropy": CrossEntropyLossLayer(),
        }
        result = switcher.get(loss_function_name, None)
        if result is None:
            raise ValueError(loss_function_name + " is an invalid loss function name")
        return result

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, ).

        Returns: 2-tuple of np.ndarray: (shuffled inputs, shuffled_targets).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # shuffle the input_dataset/target_dataset in unison
        p = np.random.permutation(len(target_dataset))

        return np.array(input_dataset[p]), np.array(target_dataset[p])

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, ).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self.checkDimensionsOfDataAndNetworkMatch(input_dataset, target_dataset)

        # iterate through the number of epochs
        for epoch_num in range(self.nb_epoch):
            # shuffling the data/targets if desired
            if self.shuffle_flag:
                input_dataset, target_dataset = self.shuffle(input_dataset, target_dataset)

            # splitting the data/targets in batches
            batch_data, batch_target = np.array_split(input_dataset, self.batch_size), np.array_split(target_dataset,
                                                                                                      self.batch_size)
            # iterate through the batches
            for batchNumber in range(len(batch_target)):
                # forward pass
                output = self.network.forward(batch_data[batchNumber])
                # calc loss and its derivative w.r.t the outputs
                self._loss_layer.forward(output, batch_target[batchNumber])
                grad_loss = self._loss_layer.backward()
                # backward pass
                self.network.backward(grad_loss)
                # 1 step gradient descent
                self.network.update_params(self.learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def checkDimensionsOfDataAndNetworkMatch(self, input_dataset, target_dataset):
        if input_dataset.shape[1] != self.network.input_dim:
            raise ValueError("Number of input dimensions of the network must match dimensions of input_dataset")
        elif target_dataset.shape[1] != self.network.neurons[-1]:
            raise ValueError("Number of output dimensions of the network must match dimensions of target_dataset")

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, ).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # forward pass
        output = self.network.forward(input_dataset)
        # calc loss
        return self._loss_layer.forward(output, target_dataset)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            - data {np.ndarray} dataset used to determined the parameters for
            the normalization.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Store 1D array of mean and std of each attribute of the data
        self.mu = np.mean(data, axis=0)
        self.sigma = np.std(data, axis=0)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            - data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Applies z-normalisation
        return (data - self.mu) / self.sigma
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def revert(self, data):
        """
        Revert the pre-processing operations to retreive the original dataset.

        Arguments:
            - data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Revert z-normalisation
        return (data * self.sigma) + self.mu
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))

if __name__ == "__main__":
    example_main()
