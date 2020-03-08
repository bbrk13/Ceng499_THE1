import numpy as np
import gzip, pickle
import sys


class MLP(object):

    def __init__(self, num_inputs=2, hidden_layers=[2], num_outputs=1):

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

        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropogation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # print("i: {} w:{}".format(i, w))
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            if i == 1:
                activations = self._sigmoid(net_inputs)
            if i == 0:
                activations = self.activation_for_hidden_layer(net_inputs)

            # save the activations for backpropogation
            self.activations[i + 1] = activations

        # return output layer activation
        return activations


    def activation_for_hidden_layer(self, X):
        X = np.where(np.absolute(X) < 1.0, (X / 2.0) + 0.5, X)
        X = np.where(X <= -1.0, 0.0, X)
        X = np.where(X >= 1.0, 1.0, X)
        # print("X: ", X)
        return X

    def derivative_activation_for_hidden_layer(self, X):
        X = np.where(np.absolute(X) < 1.0, 0.5, X)
        X = np.where(X <= -1.0, 0, X)
        X = np.where(X >= 1.0, 0, X)
        # print("derv. X ", X)
        return X



    def back_propagate(self, error):

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):
            # print("i: {} self.activations[{}] self.activations+1[{}]".format(i, self.activations[i], self.activations[i+1]))
            # get activation for previous layer
            activations = self.activations[i + 1]

            # apply sigmoid derivative function
            if i == 1 :
                delta = error * self._sigmoid_derivative(activations)
            if i == 0 :
                delta = error * self.derivative_activation_for_hidden_layer(activations)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0], -1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)

    def train(self, inputs, targets, epochs, learning_rate):
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
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate

    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _mse(self, target, output):
        return np.average((target - output) ** 2)


if __name__ == "__main__":
    with gzip.open("data1_test.pickle.gz") as f:
        data, labels = pickle.load(f, encoding='latin1')
    # create a dataset to train a network for the sum operation
    data_re = np.asarray(data, dtype=float)
    labels_re = np.reshape(labels, (200, 1))
    labels_re = labels_re * 1.0

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2, [2], 1)

    # train network
    mlp.train(data_re, labels_re, 1000, 1)
