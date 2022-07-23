import os

import matplotlib.pyplot as plt


def save_result(model_name: str, model_accuracy: str, accuracy_result: dict) -> None:
    plt.figure(0)
    plt.title('Activation Function Comparison ({})'.format(model_name))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    for key_1, value_1 in accuracy_result.items():
        for key_2, value_2 in accuracy_result[key_1].items():
            epoch_history = accuracy_result[key_1][key_2]
            plt.plot(range(1, len(epoch_history) + 1), epoch_history, label="{} ({})".format(key_2, str(key_1).lower()))
    plt.legend(loc='lower right')

    if not os.path.exists("figures"):
        os.makedirs("figures")
    filename = "{}_{}_result.png".format(str(model_name).lower(), model_accuracy)
    plt.savefig(os.path.join('figures', filename))
