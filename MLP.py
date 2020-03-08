import numpy as np
import gzip, pickle


class MLP(object):
    """A Multilayer Perceptron class.
    """
    np.random.seed(1234)

    def __init__(self, num_inputs=2, hidden_layers=[2], num_outputs=1):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs
        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """

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

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

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
            self.activations[i + 1] = activations

        # return output layer activation
        return activations

    def back_propagate(self, error):
        # error = 0.0
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]
            delta = error * self._sigmoid_derivative(activations)  # ndarray([0.1, 0.2]) --> ndarray([[0.1, 0.2]])
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]  # ndarray([0.1, 0.2]) --> ndarray([[0.1], [0.2]])
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)

            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)
        return error

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _sigmoid(self, x):
        """Sigmoid activation function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """

        y = 1.0 / (1 + np.exp(-x))
        return y

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            # print("Original W{} {}".format(i, weights))
            derivatives = self.derivatives[i]
            weights = weights + derivatives * learning_rate
            # self.weights = weights
            # print("Updated W{} {}".format(i, weights))

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_error = 0.0
            for (input, target) in zip(inputs, targets):
                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)

            # report the error
            print("Error: {} at epoch {}".format(sum_error / len(inputs), i))

    def _mse(self, target, output):
        return np.average((target - output) ** 2)


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
    epochs = 50
    for z in range(epochs):
        for i in range(0, number_of_instances, 2):
            # print(type(data))
            # print("data: {}".format(data[0]))
            # print(data.item(i))

            # set random values for network's input
            # inputs = np.random.rand(mlp.num_inputs)
            inputs = np.array([data.item(i), data.item(i + 1)], dtype=float)
            # total_inputs = np.append(total_inputs, inputs, axis=0)
            target = np.array([labels.item(int(i / 2))], dtype=float)
            # total_targets = np.append(total_targets, target, axis=0)
            # print("inputs =", inputs)

            # perform forward propagation
            output = mlp.forward_propagate(inputs)
            # print("i/2 = ", int(i/2))
            # print("labels.item(i/2) = ", labels.item(int(i/2)))
            error = labels.item(int(i / 2)) - output

            output_vector = np.append(output_vector, output)
            # print("Network inputs: {}".format(inputs))
            # print("Network activation: {}".format(output))
            # print("expected: ", labels.item(int(i/2)))
            # print("error = ", error)
            loss += error
            mlp.back_propagate(error)
            mlp.gradient_descent(learning_rate=0.1)

        loss = loss / number_of_instances
        print("loss = ", loss)
        # mlp.back_propagate(loss)
        # mlp.gradient_descent(learning_rate=0.1)
