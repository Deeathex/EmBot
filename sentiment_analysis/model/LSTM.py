from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D


class LSTM:
    def __init__(self):
        self._history = None
        self._classifier = Sequential()

    def _build_model(self):
        pass

    def _summarize_model(self):
        self._classifier.summary()

    def _train_model(self):
        pass
