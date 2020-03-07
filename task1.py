import numpy as np
import pickle, gzip


class ANN:

    def __init__(self):
        np.random.seed(1234)
        with gzip.open("data1_test.pickle.gz") as f:
            self.data, self.labels = pickle.load(f, encoding='latin1')
        np.asarray(self.data, dtype=float)
        np.asarray(self.labels, dtype=float)
        print("shape of data", self.data.shape)
        print("shape of labels", self.labels.shape)
        self.ni = self.data.shape[1]
        self.no = 1
        self.nh = self.calculate_number_of_hidden_nodes(self.ni, self.no)

        self.w1 = np.random.randn(self.data.shape[1], self.data.shape[0]) * 0.01  # 200x2 matrix
        self.w2 = np.random.randn(self.labels.shape[0], 1) * 0.01 # 200x1 matrix

        self.b1 = np.zeros((self.ni, self.nh))
        self.b2 = np.zeros((self.nh, self.no))

    def activation_for_output(self, x):
        return 1 / (1 + np.exp(-x))

    def derv_sigmoid(self, x):
        # fx = self.activation_for_output(x)
        # return fx * (1 - fx)
        # print("derv_sigmoid => shape of x =", x.shape)
        # return np.multiply(x, (1 - x))
        # return np.multiply(x, x)
        return x.T * (1 - x)

    def activation_for_hidden_layers(self, X):
        '''result = []
        for x in X:
            if x <= -1:
                result.append(0.0)
            elif -1 < x < 1:
                result.append((x / 2) + 0.5)
            elif x >= 1:
                result.append(1.0)
        return result
        '''
        np.where(X <= -1, 0.0, X)
        np.where(X >= 1, 1.0, X)
        np.where(np.absolute(X) < 1, (X / 2) + 0.5, X)
        return X

    def loss_function(self, number_of_training_examples, training_outputs, calculated_outputs):
        result = 0.0
        for i in number_of_training_examples:
            result = (training_outputs[i] - calculated_outputs[i]) ** 2
        result = result / number_of_training_examples
        return result

    def calculate_number_of_hidden_nodes(self, ni, no):
        nh = np.floor((ni + no) / 2 + 1)
        return int(nh)

    def feedForward(self, X):
        self.l1 = np.dot(X, self.w1)  # multiplying with first weight
        # print(self.l1)
        self.l2 = self.activation_for_hidden_layers(self.l1)  # activation function of hidden layer
        self.l3 = np.dot(self.l2, self.w2)  # multiplying with second weights
        result = self.activation_for_output(self.l3)
        return result

    def backPropagation(self, X, y, output):
        self.error_sum = y - output
        self.delta = self.error_sum * self.derv_sigmoid(output)

        self.l2_error = self.delta.dot(self.w2.T)
        self.l2_delta = self.l2_error * self.derv_sigmoid(self.l2)

        self.w1 += X.T.dot(self.l2_delta)
        self.w2 += self.l2.T.dot(self.delta)

    def train(self, X, y):
        output = self.feedForward(X)
        self.backPropagation(X, y, output)


if __name__ == '__main__':
    my_ann = ANN()
    for i in range(1000):
        if i % 100 == 0:
            print("Loss: " + str(np.mean(np.square(my_ann.labels - my_ann.feedForward(my_ann.data)))))
        my_ann.train(my_ann.data, my_ann.labels)
    # print("predicted output: " + str(my_ann.feedForward(my_ann.data)))
