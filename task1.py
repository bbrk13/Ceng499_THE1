import numpy as np
import pickle, gzip
from scipy.special import expit


class ANN:

    def __init__(self):
        np.random.seed(1234)
        with gzip.open("data1_test.pickle.gz") as f:
            self.data, self.labels = pickle.load(f, encoding='latin1')
        np.asarray(self.data, dtype=float)
        np.asarray(self.labels, dtype=float)
        print("shape of data", self.data.shape)
        print("first row of data = ", self.data[0][0][0])
        print("shape of labels", self.labels.shape)
        self.ni = self.data.shape[1]
        self.no = 1
        self.nh = self.calculate_number_of_hidden_nodes(self.ni, self.no)

        self.w1 = np.random.randn(self.data.shape[1], self.nh) * 0.01  # 2x2 matrix
        self.w2 = np.random.randn(self.nh, 1) * 0.01  # 2x1 matrix

        self.b1 = np.zeros((self.ni, self.nh))
        self.b2 = np.zeros((self.nh, self.no))

    def activation_for_output(self, x):
        # return 1 / (1 + np.exp(-x))
        return expit(-x)

    def elementwise_mult(self, matrix1, matrix2):
        len1 = len(matrix1)
        len2 = len(matrix1[0])
        result = np.array([], dtype=float)
        for index1 in range(len1):
            row = np.array([], dtype=float)
            for index2 in range(len2):
                tmp = matrix1[index1][index2] * matrix2[index1][index2]
                row = np.append(row, tmp)
            result = np.append(result, row, axis=0)
        return result



    def derv_sigmoid(self, x):
        # fx = self.activation_for_output(x)
        # return fx * (1 - fx)
        # print("derv_sigmoid => shape of x =", x.shape)
        # return np.multiply(x, (1 - x))
        # return np.multiply(x, x)
        return self.elementwise_mult(x, (1-x))

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
        # self.error_sum = y - output
        self.error_sum = np.subtract(y, output.T)
        self.error_sum = self.error_sum.T
        # print("shape of y", y.shape)
        # print("shape of output", output.shape)
        # delta_re = delta.reshape(delta.shape[0], -1).T
        # self.error_sum = self.error_sum.reshape(self.error_sum[0], -1)
        print("shape of error_sum", self.error_sum.shape)
        print("error_sum", self.error_sum)
        print("shape of derv_sigmoid output", self.derv_sigmoid(output).shape)
        print("derv_sigmoid_output", self.derv_sigmoid(output))
        self.delta = self.error_sum * self.derv_sigmoid(output)
        # self.delta = self.elementwise_mult(self.error_sum, self.derv_sigmoid(output))

        self.l2_error = self.delta.dot(self.w2.T)
        self.l2_delta = self.l2_error * self.derv_sigmoid(self.l2)

        self.w1 += X.T.dot(self.l2_delta)
        self.w2 += self.l2.T.dot(self.delta)

    def train(self, X, y):
        output = self.feedForward(X)
        self.backPropagation(X, y, output)


if __name__ == '__main__':
    my_ann = ANN()
    m1 = [
        [1, 2],
        [3, 4]
    ]
    m2 = [
        [5, 6],
        [7, 8]
    ]
    print(my_ann.elementwise_mult(m1, m2))
    for i in range(1000):
        if i % 100 == 0:
            print("Loss: " + str(np.mean(np.square(my_ann.labels - my_ann.feedForward(my_ann.data)))))
        my_ann.train(my_ann.data, my_ann.labels)
    # print("predicted output: " + str(my_ann.feedForward(my_ann.data)))
