import matplotlib.pyplot as plt


def show_result(x, y, prediction_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth', fontsize=18)

    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Prediction Result', fontsize=18)

    for i in range(x.shape[0]):
        if prediction_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()


def show_learning_curve(epoch_history, loss_history):
    plt.plot(epoch_history, loss_history)
    plt.title("Learning Curve")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
