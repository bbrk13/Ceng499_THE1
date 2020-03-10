import numpy as np
import gzip, pickle
import sys


class MLP(object):

    def __init__(self, num_inputs=2, hidden_layers=[2], num_outputs=1):

        np.random.seed(1234)

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        layers = [num_inputs] + hidden_layers + [num_outputs]
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def forward_propagate(self, inputs):
        activations = inputs
        self.activations[0] = activations
        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activations, w)
            if i == 1:
                activations = self.sigmoid(net_inputs)
            if i == 0:
                activations = self.activation_for_hidden_layer(net_inputs)
            self.activations[i + 1] = activations
        return activations


    def activation_for_hidden_layer(self, X):
        # print("activation_for_hidden_layer X: ", X)

        X = np.where(np.absolute(X) < 1.0, (X / 2.0) + 0.5, X)
        X = np.where(X <= -1.0, 0.0, X)
        X = np.where(X >= 1.0, 1.0, X)
        '''
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if -1.0 < X.item(i, j) < 1.0:
                    X[i][j] = X.item(i, j) / 2.0 + 0.5
                elif X.item(i, j) <= -1.0 :
                    X[i][j] = 0.0
                elif X.item(i, j) >= 1.0 :
                    X[i][j] = 1.0
        return X
        '''

        # print("after activation X: ", X)
        return X

    def derivative_activation_for_hidden_layer(self, X):
        # print("derivative_activation_for_hidden_layer X: ", X)
        X = np.where(np.absolute(X) < 1.0, 0.5, X)
        X = np.where(X <= -1.0, 0, X)
        X = np.where(X >= 1.0, 0, X)
        return X



    def back_propagate(self, error):

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]
            if i == 1 :
                delta = error * self.sigmoid_derivative(activations)
            if i == 0 :
                delta = error * self.derivative_activation_for_hidden_layer(activations)
            delta_re = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations, delta_re)
            error = np.dot(delta, self.weights[i].T)

    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_errors = 0

            for j, input in enumerate(inputs):
                target = targets[j]

                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                self.gradient_descent(learning_rate)

                sum_errors += self.mean_square_error(target, output)

            # Epoch complete, report the training error
            # print("Error: {} at epoch {}".format(sum_errors / len(inputs), i + 1))

        # print("Training complete!")
        # print("=====")

    def gradient_descent(self, learningRate=1):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate

    def sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def mean_square_error(self, target, output):
        return np.average((target - output) ** 2)

    def calculate_accuracy(self, data_test, labels_test):
        number_of_test_instances = len(labels_test)
        number_of_correct_classfies = 0
        for j, input in enumerate(data_test):
            target = labels_test[j]

            # activate the network!
            output = self.forward_propagate(input)
            if output >= 0.5:
                output = 1
            else:
                output = 0
            error = target - output
            if error == 0:
                number_of_correct_classfies += 1
        accuracy = number_of_correct_classfies / number_of_test_instances
        # print("number of test instances: ", number_of_test_instances)
        # print("number of correct classfies: ", number_of_correct_classfies)
        print(accuracy)


# if __name__ == "__main__":

train_data_path = sys.argv[1]
test_data_path = sys.argv[2]
epochs = int(sys.argv[3])
with gzip.open(train_data_path) as f:
    data, labels = pickle.load(f, encoding='latin1')
number_of_instances = labels.shape[0]
data_re = np.asarray(data, dtype=float)
labels_re = np.reshape(labels, (number_of_instances, 1))
labels_re = labels_re * 1.0
ni = data.shape[1]
no = 1
nh = np.floor((ni + no) / 2 + 1)
nh = int(nh)
# print(type(nh))

with gzip.open(test_data_path) as f1:
    test_data, test_labels = pickle.load(f1, encoding='latin1')

number_of_instances_test = test_labels.shape[0]
data_re_test = np.asarray(test_data, dtype=float)
labels_re_test = np.reshape(test_labels, (number_of_instances_test, 1))
labels_re_test = labels_re_test * 1.0

mlp = MLP(ni, [nh], no)

mlp.train(data_re, labels_re, epochs, 0.2)

mlp.calculate_accuracy(test_data, test_labels)
