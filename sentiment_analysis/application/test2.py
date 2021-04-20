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
from sentiment_analysis.utils.constants import DATASET_PATH, TEXT, TOKENIZED_SENTENCE, BERT_EMBEDDING

import nltk

nltk.download('stopwords')
nltk.download('wordnet')

nltk.download('punkt')


def get_lemm_text(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


stop_words = stopwords.words('english')

data_frame = pd.read_csv('../outputs/data_frame.csv', sep=',')

EMBEDDING_DIMENSION = 64
VOCABULARY_SIZE = 2000
MAX_LENGTH = 100
OOV_TOK = '<OOV>'
TRUNCATE_TYPE = 'post'
PADDING_TYPE = 'post'

emotion_label = data_frame.emotion.factorize()
print(data_frame[BERT_EMBEDDING][0][0])
print(data_frame[BERT_EMBEDDING][1][0])
padded_sequence = data_frame.apply(lambda row: pad_sequences(row[BERT_EMBEDDING][0], maxlen=MAX_LENGTH), axis=1)
print(padded_sequence)

model = Sequential()
model.add(Embedding(len(MAX_LENGTH) + 1, EMBEDDING_DIMENSION))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(EMBEDDING_DIMENSION, dropout=0.3, recurrent_dropout=0.3)))
model.add(Dense(EMBEDDING_DIMENSION, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(EMBEDDING_DIMENSION, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(100))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 10
history = model.fit(padded_sequence, emotion_label[0], epochs=num_epochs)
