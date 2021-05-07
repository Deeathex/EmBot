from sentiment_analysis.model.ArtificialNeuralNetwork import ArtificialNeuralNetwork

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
from tensorflow.keras.layers import Dropout

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from sentiment_analysis.nlp.NLPModule import NLPModule
from sentiment_analysis.utils.constants import PROCESSED_SENTENCE

import time

EPOCHS = 100
STEPS_PER_EPOCH = 25
BATCH_SIZE = 120
VALIDATION_STEPS = 10


class LongShortThermMemoryNetwork(ArtificialNeuralNetwork):
    def __init__(self, ):
        super().__init__()
        self.__embedding_dim = 100
        self.nlp_module = NLPModule(10000, '../dataset/emotions_dataset')

        self.__X_train = None
        self.__y_train = None
        self.__X_val = None
        self.__y_val = None

    def build(self):
        self._build_model()
        self._summarize_model()

    def train(self):
        self._build_model()
        self._train_model()
        self._save_metrics_and_model()

    def _build_model(self):
        train_df = self.nlp_module.get_train_dataframe()
        val_df = self.nlp_module.get_validation_dataframe()
        train_df['length'] = train_df.sentence.apply(lambda x: len(x))
        MAXLEN = train_df.length.max()
        X_train_pad = self.__get_padded_data(train_df, MAXLEN)
        X_val_pad = self.__get_padded_data(val_df, MAXLEN)

        y_train = to_categorical(train_df.emotion.values)
        y_val = to_categorical(val_df.emotion.values)

        self.__X_train = X_train_pad
        self.__y_train = y_train
        self.__X_val = X_val_pad
        self.__y_val = y_val

        self._classifier = Sequential()
        self._classifier.add(Embedding(input_dim=self.nlp_module.get_vocabulary_size(),  # the whole vocabulary size
                                       output_dim=self.__embedding_dim,  # vector space dimension
                                       input_length=X_train_pad.shape[1],  # max_len of text sequence
                                       weights=[self.nlp_module.compute_word2vec_weight_matrix()],
                                       trainable=False))
        self._classifier.add(Dropout(0.2))
        self._classifier.add(Bidirectional(LSTM(100, return_sequences=True)))
        self._classifier.add(Dropout(0.2))
        self._classifier.add(Bidirectional(LSTM(200, return_sequences=True)))
        self._classifier.add(Dropout(0.2))
        self._classifier.add(Bidirectional(LSTM(100, return_sequences=False)))
        self._classifier.add(Dense(units=6, activation='softmax'))
        self._classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

    def _train_model(self):
        print('Training the network..')
        self._history = self._classifier.fit(self.__X_train, self.__y_train,
                                             epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, batch_size=BATCH_SIZE,
                                             validation_data=(self.__X_val, self.__y_val),
                                             validation_steps=VALIDATION_STEPS,
                                             verbose=1)

    def __get_padded_data(self, dataframe, max_len):
        X_data = self.nlp_module.get_tokenizer().texts_to_sequences(dataframe[PROCESSED_SENTENCE])
        return pad_sequences(X_data, maxlen=max_len, padding='post')


lstm = LongShortThermMemoryNetwork()
# lstm.build()
start = time.time()
lstm.train()
done = time.time()
elapsed = done - start
print('Duration in seconds: ', elapsed)
