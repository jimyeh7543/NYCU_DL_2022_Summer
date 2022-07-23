import math

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)


def relu(x):
    return np.maximum(0.0, x)


def derivative_relu(x):
    return np.heaviside(x, 0.0)


def tanh(x):
    return np.tanh(x)


def derivative_tanh(x):
    return 1.0 - x**2


def leaky_relu(x):
    return np.maximum(0.0, x) + 0.01 * np.minimum(0.0, x)


def derivative_leaky_relu(x):
    x[x > 0.0] = 1.0
    x[x <= 0.0] = 0.01
    return x


def cross_entropy_loss(prediction_y, y):
    n = prediction_y.shape[1]
    cost = -1 / n * (np.dot(y.T, np.log(prediction_y)) +
                     np.dot(1 - y.T, np.log(1 - prediction_y)))
    return np.squeeze(cost)


def cross_entropy_derivative_loss(prediction_y, y):
    return -np.divide(y, prediction_y) + np.divide(1 - y, 1 - prediction_y)


def mse_loss(prediction_y, y):
    return np.mean((prediction_y - y)**2)


def mse_derivative_loss(prediction_y, y):
    return 2 * (prediction_y - y) / len(y)


def without_activation_function(x):
    return x


def convert_probabilities_to_class(probabilities):
    prediction_y = np.copy(probabilities)
    prediction_y[prediction_y > 0.5] = 1
    prediction_y[prediction_y <= 0.5] = 0
    return prediction_y
