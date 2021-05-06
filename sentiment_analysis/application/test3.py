import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

import gensim.downloader as api

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
from tensorflow.keras.layers import Dropout

import datetime
import csv
import pickle

sns.set()

METRICS_ = '../metrics/'

train_df = pd.read_csv('../dataset/emotions_dataset/train.txt', sep='|')
test_df = pd.read_csv('../dataset/emotions_dataset/test.txt', sep='|')
val_df = pd.read_csv('../dataset/emotions_dataset/val.txt', sep='|')

train_df.columns = ['sentence', 'emotion']
test_df.columns = ['sentence', 'emotion']
val_df.columns = ['sentence', 'emotion']

# EDA
sns.countplot(train_df.emotion)
# plt.show()

train_df['length'] = train_df.sentence.apply(lambda x: len(x))
plt.plot(train_df.length)
# plt.show()
print('Max length of our text body: ', train_df.length.max())

# text pre processing
print(train_df.head())
labeled_dict = {'joy': 0, 'anger': 1, 'love': 2, 'sadness': 3, 'fear': 4, 'surprise': 5}
train_df['emotion'] = train_df.emotion.replace(labeled_dict)
test_df['emotion'] = test_df.emotion.replace(labeled_dict)
val_df['emotion'] = val_df.emotion.replace(labeled_dict)

num_words = 10000  # this means 15000 unique words can be taken
tokenizer = Tokenizer(num_words, lower=True)
tokenizer.fit_on_texts(train_df.sentence.values)

# this is whole unique words in our corpus
# but we are taking only 10000 words in our model
MAXLEN = train_df.length.max()
X_train = tokenizer.texts_to_sequences(train_df['sentence'])
X_train_pad = pad_sequences(X_train, maxlen=MAXLEN, padding='post')
X_test = tokenizer.texts_to_sequences(test_df.sentence)
X_test_pad = pad_sequences(X_test, maxlen=MAXLEN, padding='post')
X_val = tokenizer.texts_to_sequences(val_df.sentence)
X_val_pad = pad_sequences(X_val, maxlen=MAXLEN, padding='post')

y_train = to_categorical(train_df.emotion.values)
y_test = to_categorical(test_df.emotion.values)
y_val = to_categorical(val_df.emotion.values)

glove_gensim = api.load('glove-wiki-gigaword-100')  # 100 dimension
print(glove_gensim['cat'].shape[0])

# creation of word2vec weight matrix
vector_size = 100
gensim_weight_matrix = np.zeros((num_words, vector_size))
gensim_weight_matrix.shape

for word, index in tokenizer.word_index.items():
    if index < num_words:  # since index starts with zero
        if word in glove_gensim.wv.vocab:
            gensim_weight_matrix[index] = glove_gensim[word]
        else:
            gensim_weight_matrix[index] = np.zeros(100)

# Designing a LSTM model with GPU support
# EMBEDDING_DIM = 100  # this means the embedding layer will create  a vector in 100 dimension
# model = Sequential()
# model.add(Embedding(input_dim=num_words,  # the whole vocabulary size
#                     output_dim=EMBEDDING_DIM,  # vector space dimension
#                     input_length=X_train_pad.shape[1],  # max_len of text sequence
#                     weights=[gensim_weight_matrix], trainable=False))
# model.add(Dropout(0.2))
# model.add(Bidirectional(CuDNNLSTM(100, return_sequences=True)))
# model.add(Dropout(0.2))
# model.add(Bidirectional(CuDNNLSTM(200, return_sequences=True)))
# model.add(Dropout(0.2))
# model.add(Bidirectional(CuDNNLSTM(100, return_sequences=False)))
# model.add(Dense(6, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

EMBEDDING_DIM = 100  # this means the embedding layer will create  a vector in 100 dimension
model = Sequential()
model.add(Embedding(input_dim=num_words,  # the whole vocabulary size
                    output_dim=EMBEDDING_DIM,  # vector space dimension
                    input_length=X_train_pad.shape[1],  # max_len of text sequence
                    weights=[gensim_weight_matrix], trainable=False))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(200, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(100, return_sequences=False)))
model.add(Dense(6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

# EMBEDDING_DIMENSION = 64
# model = Sequential()
# model.add(Embedding(vocab_size, EMBEDDING_DIMENSION))
# model.add(SpatialDropout1D(0.3))
# model.add(Bidirectional(LSTM(EMBEDDING_DIMENSION, dropout=0.3, recurrent_dropout=0.3)))
# model.add(Dense(EMBEDDING_DIMENSION, activation='relu'))
# model.add(Dropout(0.8))
# model.add(Dense(EMBEDDING_DIMENSION, activation='relu'))
# model.add(Dropout(0.8))
# model.add(Dense(1))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
# mc = ModelCheckpoint('./model_test3.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# history_embedding = model.fit(X_train_pad, y_train,
#                               epochs=25, batch_size=120,
#                               validation_data=(X_val_pad, y_val),
#                               verbose=1, callbacks=[es, mc])


history_embedding = model.fit(X_train_pad, y_train,
                              epochs=1, batch_size=300,
                              validation_data=(X_val_pad, y_val),
                              verbose=1)

now = datetime.datetime.now()
date_index = now.strftime("%Y-%m-%d_%H-%M-%S")

print('Saving the metrics and model..')
w = csv.writer(open(METRICS_ + 'history_metrics_' + date_index + ".csv", "w"))
for key, val in history_embedding.history.items():
    w.writerow([key, val])

with open(METRICS_ + 'train_history_dict_' + date_index + '.txt', 'wb') as file_pi:
    pickle.dump(history_embedding.history, file_pi)

print('Saving the model')
model.save(METRICS_ + 'saved_model_' + date_index + '.h5')

plt.plot(history_embedding.history['accuracy'], c='b', label='train accuracy')
plt.plot(history_embedding.history['val_accuracy'], c='r', label='validation accuracy')
plt.legend(loc='lower right')
plt.show()


def get_key(value):
    for key, val in labeled_dict.items():
        if (val == value):
            return key


def predict(sentence):
    sentence_lst = []
    sentence_lst.append(sentence)
    sentence_seq = tokenizer.texts_to_sequences(sentence_lst)
    sentence_padded = pad_sequences(sentence_seq, maxlen=300, padding='post')
    ans = get_key(model.predict_classes(sentence_padded))
    print("The emotion predicted is", ans)


predict('What a wonderful day!')
