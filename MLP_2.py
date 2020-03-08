import numpy as np
import gzip, pickle
from random import random


class MLP(object):
    """A Multilayer Perceptron class.
    """

    def __init__(self, num_inputs=2, hidden_layers=[2], num_outputs=1):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs
        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """
        np.random.seed(1234)

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # save derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals.
        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """

        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropogation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

            # save the activations for backpropogation
            self.activations[i + 1] = activations

        # return output layer activation
        return activations

    def back_propagate(self, error):
        """Backpropogates an error signal.
        Args:
            error (ndarray): The error to backprop.
        Returns:
            error (ndarray): The final error of the input
        """

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):
            # get activation for previous layer
            activations = self.activations[i + 1]

            # apply sigmoid derivative function
            delta = error * self._sigmoid_derivative(activations)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0], -1)
            # print("current_activations", current_activations)
            # print("delta_re", delta_re)
            # print("delta", delta)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)

    def train(self, inputs, targets, epochs, learning_rate):
        """Trains model running forward prop and backprop
        Args:
            inputs (ndarray): X
            targets (ndarray): Y
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
        """
        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0

            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # activate the network!
                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                # now perform gradient descent on the derivatives
                # (this will update the weights
                self.gradient_descent(learning_rate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)

            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(sum_errors / len(inputs), i + 1))

        print("Training complete!")
        print("=====")

    def gradient_descent(self, learningRate=1):
        """Learns by descending the gradient
        Args:
            learningRate (float): How fast to learn.
        """
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate

    def _sigmoid(self, x):
        """Sigmoid activation function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """

        y = 1.0 / (1 + np.exp(-x))
        return y

    def _sigmoid_derivative(self, x):
        """Sigmoid derivative function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """
        return x * (1.0 - x)

    def _mse(self, target, output):
        """Mean Squared Error loss function
        Args:
            target (ndarray): The ground trut
            output (ndarray): The predicted values
        Returns:
            (float): Output
        """
        return np.average((target - output) ** 2)


if __name__ == "__main__":
    with gzip.open("data1_test.pickle.gz") as f:
        data, labels = pickle.load(f, encoding='latin1')
    # create a dataset to train a network for the sum operation
    items = np.array([[random() / 2 for _ in range(2)] for _ in range(200)])
    data_re = np.asarray(data, dtype=float)
    print("shape of inputs = ", items.shape)
    # print("inputs", items)
    targets = np.array([[i[0] + i[1]] for i in items])
    print("shape of targets = ", targets.shape)
    # print("targets", targets)
    print("shape of data_re = ", data_re.shape)
    print("shape of labels = ", labels.shape)
    labels_re = np.reshape(labels, (200, 1))

    print("shape of labels_re", labels_re.shape)
    print(data_re.shape == items.shape)
    print(targets.shape == labels_re.shape)
    print("type of data", type(data))
    print("type of data_re", type(data_re))
    print("type of items", type(items))
    print("data_re = ", type(data_re.item(0)))
    print("inputs = ", type(items.item(0)))
    print("data_re = ", data_re)
    print("inputs", items)
    labels_re = labels_re * 1.0
    # print("labels", labels)
    # print("labels_re", labels_re)
    # print("targets", targets)

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2, [2], 1)

    # train network
    mlp.train(data_re, labels_re, 100, 0.1)

    # create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    # get a prediction
    output = mlp.forward_propagate(input)

    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))
    '''
if __name__ == "__main__":
    # create a Multilayer Perceptron
    mlp = MLP()
    with gzip.open("data1_test.pickle.gz") as f:
        data, labels = pickle.load(f, encoding='latin1')
    number_of_instances = data.shape[0]
    print(number_of_instances)
    output_vector = np.array([], dtype=float)
    loss = 0.0
    total_inputs = np.array([], dtype=float)
    total_targets = np.array([], dtype=float)
    epochs = 500
    for z in range(epochs):
        for i in range(0, number_of_instances, 2):
            # print(type(data))
            # print("data: {}".format(data[0]))
            # print(data.item(i))

            # set random values for network's input
            # inputs = np.random.rand(mlp.num_inputs)
            inputs = np.array([data.item(i), data.item(i + 1)], dtype=float)
            # total_inputs = np.append(total_inputs, inputs, axis=0)
            target = np.array([labels.item(int(i/2))], dtype=float)
            # total_targets = np.append(total_targets, target, axis=0)
            # print("inputs =", inputs)

            # perform forward propagation
            output = mlp.forward_propagate(inputs)
            # print("i/2 = ", int(i/2))
            # print("labels.item(i/2) = ", labels.item(int(i/2)))
            error = labels.item(int(i/2)) - output

            output_vector = np.append(output_vector, output)
            # print("Network inputs: {}".format(inputs))
            # print("Network activation: {}".format(output))
            # print("expected: ", labels.item(int(i/2)))
            # print("error = ", error)
            loss += error
            mlp.back_propagate(error)
            mlp.gradient_descent(learningRate=0.1)

        loss = loss / number_of_instances
        print("loss = ", loss)
        # mlp.back_propagate(loss)
        # mlp.gradient_descent(learning_rate=0.1)'''
