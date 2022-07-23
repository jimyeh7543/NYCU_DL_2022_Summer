from utils import *
from visualization import show_result, show_learning_curve
from simple_neural_network import SimpleNeuralNetwork
from input_generator import generate_linear, generate_xor_easy

if __name__ == '__main__':
    nn_architecture = [{
        "in_features": 2,
        "out_features": 128,
        "activation_function": "sigmoid"
    }, {
        "in_features": 128,
        "out_features": 256,
        "activation_function": "sigmoid"
    }, {
        "in_features": 256,
        "out_features": 1,
        "activation_function": "sigmoid"
    }]
    n_epochs = 10000
    x, y = generate_linear(n=100)
    #x, y = generate_xor_easy()

    simple_nn = SimpleNeuralNetwork(nn_architecture,
                                    optimizer="adam",
                                    learning_rate=0.01,
                                    convolution=False)
    loss_history = []
    epoch_history = []
    for epoch in range(1, n_epochs + 1):
        prediction_y = simple_nn.forward(x)
        loss = cross_entropy_loss(prediction_y, y)
        simple_nn.backward(cross_entropy_derivative_loss(prediction_y, y))
        simple_nn.update()
        loss_history.append(loss)
        epoch_history.append(epoch)
        prediction_labels = convert_probabilities_to_class(prediction_y)
        accuracy = float(np.sum(prediction_labels == y)) / len(y)
        print(f'Epoch: {epoch}, loss: {loss}, Accuracy: {accuracy:}')
        if accuracy == 1:
            break

    show_learning_curve(epoch_history, loss_history)
    show_result(x, y, prediction_labels)
