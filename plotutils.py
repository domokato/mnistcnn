import matplotlib.pyplot as plt


def plot_accuracy(history, title, validation_set_label):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.tick_params(axis='y', which='both', labelleft=True, labelright=True)
    plt.legend(['Training set', validation_set_label], loc='right')
    plt.show()
