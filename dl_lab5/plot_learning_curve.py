import re
import json
import matplotlib.pyplot as plt


def read_training_history():
    training_history = []
    index = -1
    with open("dl_lab5.log", 'r') as f:
        for line in f:
            if re.match(r'^INFO:dl_lab5:Namespace', line) or re.match(r'^INFO:dl_lab5:Highest', line):
                continue

            dictionary_part = re.search('^INFO:dl_lab5:(.*)', line).group(1)
            data = eval(dictionary_part)
            training_history.append(data)
            index += 1
    return training_history


def plot_result(training_history):
    epoch = [dict['epoch'] for dict in training_history if 'epoch' in dict]
    loss_d = [dict['loss_d'] for dict in training_history if 'loss_d' in dict]
    loss_g = [dict['loss_g'] for dict in training_history if 'loss_g' in dict]
    test_acc = [dict['test_acc'] for dict in training_history if 'test_acc' in dict]
    new_test_acc = [dict['new_test_acc'] for dict in training_history if 'new_test_acc' in dict]
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0,1.1])
    lns = ax1.plot(epoch, loss_d, label="Discriminator Loss")
    lns += ax1.plot(epoch, loss_g, label="Generator Loss")
    lns += ax2.plot(epoch, test_acc, "r", label="ACC on test.json")
    lns += ax2.plot(epoch, new_test_acc, "m", label="ACC on new_test.json")
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")
    ax1.grid()
    plt.show()


def main():
    training_history = read_training_history()
    plot_result(training_history)


if __name__ == '__main__':
    main()
