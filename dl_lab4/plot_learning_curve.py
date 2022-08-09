import json
import matplotlib.pyplot as plt


def read_training_history():
    training_history = []
    index = -1
    with open("train_record.txt", 'r') as f:
        for line in f:
            if line.startswith("args"):
                continue
            data = json.loads(line.strip())
            if data["epoch"] != index:
                training_history.append(data)
                index += 1
            else:
                training_history[index].update(data)
    return training_history


def plot_result(training_history):
    epoch = [dict['epoch'] for dict in training_history if 'epoch' in dict]
    tfr = [dict['tfr'] for dict in training_history if 'tfr' in dict]
    kl_weight = [dict['kl_weight'] for dict in training_history if 'kl_weight' in dict]
    loss = [dict['loss'] for dict in training_history if 'loss' in dict]
    kld_loss = [dict['kld_loss'] for dict in training_history if 'kld_loss' in dict]
    mse_loss = [dict['mse_loss'] for dict in training_history if 'mse_loss' in dict]
    validation_psnr = [dict['validation_psnr'] for dict in training_history if 'validation_psnr' in dict]
    test_psnr = [dict['test_psnr'] for dict in training_history if 'test_psnr' in dict]
    epoch_psnr = [dict['epoch'] for dict in training_history if 'validation_psnr' in dict]
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel("Score / Weight")
    ax1.set_ylim([0, 1.1])
    ax2 = ax1.twinx()
    ax2.set_ylabel("PSNR Score")
    ax2.set_ylim([min(validation_psnr) - 1, max(validation_psnr) + 1])
    lns = ax1.plot(epoch, tfr, "b", label="TFR", linestyle="-.")
    lns += ax1.plot(epoch, kl_weight, "g", label="KL Weight", linestyle="-.")
    lns += ax1.plot(epoch, loss, label="Loss")
    lns += ax1.plot(epoch, kld_loss, label="KLD Loss")
    lns += ax1.plot(epoch, mse_loss, label="MSE Loss")
    lns += ax2.plot(epoch_psnr, validation_psnr, "r.", label="Validation PSNR")
    lns += ax2.plot(epoch_psnr, test_psnr, "m.", label="Test PSNR")
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")
    ax1.grid()
    plt.show()


def main():
    training_history = read_training_history()
    plot_result(training_history)


if __name__ == '__main__':
    main()
