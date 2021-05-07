import pickle
import matplotlib.pyplot as plot
from tensorflow.keras.models import load_model
from tensorflow.python.keras.utils.vis_utils import plot_model


class Metrics:
    def __init__(self, model_filename, history_filename):
        self.__classifier = load_model(model_filename)
        self.__history = pickle.load(open(history_filename, "rb"))

    def plot_model(self, filename_to_plot_model):
        plot_model(self.__classifier, to_file=filename_to_plot_model,
                   show_shapes=True,
                   show_layer_names=True)

    def summarize_history_for_accuracy(self, plot_filename=None):
        # list all data in history
        print(self.__history.keys())
        # https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
        # summarize history for accuracy
        plot.plot(self.__history['accuracy'])
        plot.plot(self.__history['val_accuracy'])
        plot.title('Model accuracy')
        plot.ylabel('Accuracy')
        plot.xlabel('Epoch')
        plot.legend(['Train', 'Validation'], loc='upper left')
        if plot_filename is not None:
            plot.savefig(plot_filename, bbox_inches='tight')
        plot.show()

    def summarize_history_for_loss(self, plot_filename=None):
        # summarize history for loss
        plot.plot(self.__history['loss'])
        plot.plot(self.__history['val_loss'])
        plot.title('Model loss')
        plot.ylabel('Loss')
        plot.xlabel('Epoch')
        plot.legend(['Train', 'Validation'], loc='upper left')
        if plot_filename is not None:
            plot.savefig(plot_filename, bbox_inches='tight')
        plot.show()

    def print_accuracy(self):
        print(self.__history['accuracy'])
        print(self.__history['val_accuracy'])

    def print_loss(self):
        print(self.__history['loss'])
        print(self.__history['val_loss'])


metrics = Metrics('saved_model_2021-05-07_14-28-54.h5', 'train_history_dict_2021-05-07_14-28-54.txt')
metrics.plot_model('plot_model.png')
metrics.summarize_history_for_accuracy('model_accuracy_plot.png')
metrics.summarize_history_for_loss('model_loss_plot.png')
metrics.print_accuracy()
metrics.print_loss()
