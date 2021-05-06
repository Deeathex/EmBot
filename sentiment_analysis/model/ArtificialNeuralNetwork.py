import datetime
import csv
import pickle

from keras.utils.vis_utils import plot_model

METRICS_ = '../metrics/'


class ArtificialNeuralNetwork:
    def __init__(self):
        self._history = None
        self._classifier = None

    def _build_model(self):
        pass

    def _summarize_model(self):
        self._classifier.summary()
        plot_model(self._classifier, to_file='../outputs/model_lstm_plot.png', show_shapes=True, show_layer_names=True)

    def _train_model(self):
        pass

    def _save_metrics_and_model(self):
        now = datetime.datetime.now()
        date_index = now.strftime("%Y-%m-%d_%H-%M-%S")

        print('Saving the metrics and model..')
        w = csv.writer(open(METRICS_ + 'history_metrics_' + date_index + ".csv", "w"))
        for key, val in self._history.history.items():
            w.writerow([key, val])

        with open(METRICS_ + 'train_history_dict_' + date_index + '.txt', 'wb') as file_pi:
            pickle.dump(self._history.history, file_pi)

        print('Saving the model')
        self._classifier.save(METRICS_ + 'saved_model_' + date_index + '.h5')
