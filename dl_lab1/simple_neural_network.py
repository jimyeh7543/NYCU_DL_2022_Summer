import numpy as np

from utils import *


class SimpleNeuralNetwork:

    def __init__(self,
                 nn_architecture,
                 optimizer="gd",
                 learning_rate=0.01,
                 convolution=True):
        self.nn_architecture = nn_architecture
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.convolution = convolution
        self.inputs = {}
        self.weights = {}
        self.outputs = {}
        self.backward_gradients = {}

        # variable for optimizer
        self.momentum = {}
        self.sum_of_squares_of_gradients = {}
        self.moving_average_m = {}
        self.moving_average_v = {}
        self.update_times = 1

        # init convolution filter
        self.kernel_size = 3
        self.filter = np.random.uniform(-1, 1, size=self.kernel_size)
        self.filter_backward_gradients = np.zeros(self.kernel_size)

        for i, layer in enumerate(nn_architecture):
            self.weights[i] = np.random.uniform(-1,
                                                1,
                                                size=(layer["in_features"],
                                                      layer["out_features"]))
            # Initialize variable for optimizer
            self.momentum[i] = np.zeros(
                (layer["in_features"], layer["out_features"]))
            self.sum_of_squares_of_gradients[i] = np.zeros(
                (layer["in_features"], layer["out_features"]))
            self.moving_average_m[i] = np.zeros(
                (layer["in_features"], layer["out_features"]))
            self.moving_average_v[i] = np.zeros(
                (layer["in_features"], layer["out_features"]))

    def forward(self, output):
        if self.convolution:
            self.inputs["convolution_layer_input"] = output
            output = self.convolution_forward(output)

        for i, layer in enumerate(self.nn_architecture):
            if "relu" == layer["activation_function"]:
                activation_function = relu
            elif "sigmoid" == layer["activation_function"]:
                activation_function = sigmoid
            elif "tanh" == layer["activation_function"]:
                activation_function = tanh
            elif "leaky_relu" == layer["activation_function"]:
                activation_function = leaky_relu
            else:
                activation_function = without_activation_function

            self.inputs[i] = output
            output = activation_function(np.matmul(output, self.weights[i]))
            self.outputs[i] = output
        return output

    def backward(self, derivative_loss):
        for i, layer in reversed(list(enumerate(self.nn_architecture))):
            if "relu" == layer["activation_function"]:
                derivative_activation_function = derivative_relu
            elif "sigmoid" == layer["activation_function"]:
                derivative_activation_function = derivative_sigmoid
            elif "tanh" == layer["activation_function"]:
                derivative_activation_function = derivative_tanh
            elif "leaky_relu" == layer["activation_function"]:
                derivative_activation_function = derivative_leaky_relu
            else:
                derivative_activation_function = without_activation_function

            temp = np.multiply(derivative_activation_function(self.outputs[i]),
                               derivative_loss)
            self.backward_gradients[i] = np.matmul(self.inputs[i].T, temp)
            derivative_loss = np.matmul(temp, self.weights[i].T)

        if self.convolution:
            self.convolution_backward(derivative_loss)

    def update(self):
        for i, layer in enumerate(self.nn_architecture):
            if self.optimizer == 'momentum':
                self.momentum[i] = 0.9 * self.momentum[
                    i] - self.learning_rate * self.backward_gradients[i]
                delta_weight = self.momentum[i]
            elif self.optimizer == 'adagrad':
                self.sum_of_squares_of_gradients[i] += np.square(
                    self.backward_gradients[i])
                delta_weight = -self.learning_rate * self.backward_gradients[
                    i] / np.sqrt(self.sum_of_squares_of_gradients[i] + 1e-8)
            elif self.optimizer == 'adam':
                self.moving_average_m[i] = 0.9 * self.moving_average_m[
                    i] + 0.1 * self.backward_gradients[i]
                self.moving_average_v[i] = 0.999 * self.moving_average_v[
                    i] + 0.001 * np.square(self.backward_gradients[i])
                bias_correction_m = self.moving_average_m[i] / (
                        1.0 - 0.9 ** self.update_times)
                bias_correction_v = self.moving_average_v[i] / (
                        1.0 - 0.999 ** self.update_times)
                self.update_times += 1
                delta_weight = -self.learning_rate * bias_correction_m / (
                        np.sqrt(bias_correction_v) + 1e-8)
            else:
                # Gradient descent
                delta_weight = -self.learning_rate * self.backward_gradients[i]

            self.weights[i] += delta_weight

        if self.convolution:
            self.convolution_update()

    def convolution_forward(self, output, stride=1, padding=1):
        result = np.zeros((output.shape[0], output.shape[1]))
        for i in range(0, output.shape[0]):
            temp = np.pad(output[i], padding)
            for j in range(0, output.shape[1]):
                start_index = j * stride
                result[i][j] = np.dot(
                    temp[start_index:start_index + self.kernel_size],
                    self.filter)
        return result

    def convolution_backward(self, derivative_loss):
        convolution_layer_input = self.inputs["convolution_layer_input"]
        for i in range(0, self.inputs["convolution_layer_input"].shape[0]):
            self.filter_backward_gradients[
                0] += derivative_loss[i][1] * convolution_layer_input[i][0]
            self.filter_backward_gradients[
                1] += derivative_loss[i][0] * convolution_layer_input[i][0]
            self.filter_backward_gradients[
                1] += derivative_loss[i][1] * convolution_layer_input[i][1]
            self.filter_backward_gradients[
                2] += derivative_loss[i][0] * convolution_layer_input[i][1]

    def convolution_update(self):
        for i in range(0, self.kernel_size):
            # Gradient descent
            self.filter[
                i] += -self.learning_rate * self.filter_backward_gradients[i]
