import os
import ctypes

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


def save_result(model_name: str, model_acc: str, acc_history: dict) -> None:
    plt.figure(0)
    plt.title('Comparison ({})'.format(model_name))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    for key_1, value_1 in acc_history.items():
        for key_2, value_2 in acc_history[key_1].items():
            epoch_history = acc_history[key_1][key_2]
            plt.plot(range(1, len(epoch_history) + 1), epoch_history, label="{} ({})".format(key_2, str(key_1).lower()))
    plt.legend(loc='lower right')

    if not os.path.exists("figures"):
        os.makedirs("figures")
    filename = "{}_{}_result.png".format(str(model_name).lower(), model_acc)
    plt.savefig(os.path.join('figures', filename))


def save_confusion_matrix(model_name: str, model_acc: str, y_true: list, y_pred: list, class_names: list) -> None:
    # Plot non-normalized confusion matrix
    normalization_options = [
        ("Confusion matrix, without normalization", None, "without_normalization"),
        ("Normalized confusion matrix", "true", "normalized"),
    ]
    if not os.path.exists("figures"):
        os.makedirs("figures")
    for title, normalize, normalize_str in normalization_options:
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=class_names, cmap=plt.cm.Blues, normalize=normalize)
        plt.title(title)
        filename = "{}_{}_{}.png".format(model_name.lower(), model_acc, normalize_str)
        plt.savefig(os.path.join('figures', filename))
