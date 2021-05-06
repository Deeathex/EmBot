import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sentence_transformers import SentenceTransformer
from sentiment_analysis.utils.Utils import Utils
from sentiment_analysis.utils.constants import DATASET_CSV_PATH, TEXT, TOKENIZED_SENTENCE, BERT_EMBEDDING

import nltk

nltk.download('stopwords')
nltk.download('wordnet')

nltk.download('punkt')


def get_lemm_text(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


def transform_to_list_of_floats(dataframe_row):
    l = [x.strip(' []\n') for x in dataframe_row.split(' ')]
    without_empty_strings = [string for string in l if string != ""]
    # list_of_floats = [float(i) for i in without_empty_strings]
    return without_empty_strings


def transform_to_float(list):
    print("ROW: ")
    print(list)
    list_of_floats = [float(i) for i in list]
    list_of_floats = np.asarray(list_of_floats)
    return list_of_floats


stop_words = stopwords.words('english')

# data_frame = pd.read_csv('../outputs/data_frame_test2.csv', sep=',')
data_frame = pd.read_parquet('../outputs/data_frame.pa')

EMBEDDING_DIMENSION = 64
VOCABULARY_SIZE = 2000
MAX_LENGTH = 100
OOV_TOK = '<OOV>'
TRUNCATE_TYPE = 'post'
PADDING_TYPE = 'post'

emotion_label = data_frame.emotion.factorize()
# print(data_frame[BERT_EMBEDDING][0])
#
#
#
#
# data_frame[BERT_EMBEDDING] = data_frame.apply(lambda row: transform_to_list_of_floats(row[BERT_EMBEDDING]), axis=1)
# data_frame[BERT_EMBEDDING] = data_frame.apply(lambda row: transform_to_float(row[BERT_EMBEDDING]), axis=1)
# pd = pad_sequences(data_frame[BERT_EMBEDDING][0], maxlen=MAX_LENGTH)
# print(pd)

# def test(row):
#     print(row)
#     return pad_sequences(row, dtype=float, maxlen=MAX_LENGTH)

# padded_sequence = data_frame.apply(lambda row: test(row[BERT_EMBEDDING]), axis=1)
# print(data_frame)
padded_sequence = pad_sequences(data_frame[BERT_EMBEDDING], dtype=float)
# print(padded_sequence.shape)
# print(padded_sequence)


EMBEDDING_DIMENSION = 128
model = Sequential()
model.add(Embedding(len(padded_sequence[0]) + 1, EMBEDDING_DIMENSION))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(EMBEDDING_DIMENSION, dropout=0.3, recurrent_dropout=0.3)))
model.add(Dense(EMBEDDING_DIMENSION, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(EMBEDDING_DIMENSION, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Am ajuns aici")
num_epochs = 50
history = model.fit(padded_sequence, emotion_label[0], epochs=num_epochs)
