import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Activation
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Embedding
from keras.regularizers import l2
from keras.constraints import maxnorm

MAXLEN_SEQ = 60

df = pd.read_csv('../dataset/ISEAR.csv', sep='|')
df.text = df.text.astype(str)
#       select relavant columns
isear_df = df[['emotion', 'text']]

#       code to see histogram for word count in each row -> needed for selecting the max length for padding
# isear_df['word_count'] = isear_df.text.apply(lambda x: len(str(x).split(' ')))
# print(isear_df.head(100))
# isear_df['word_count'].hist(bins=10)
# plt.show()

#       convert emotion to numeric
emotion_label = isear_df.emotion.factorize()
# (array([0, 1, 2, ..., 4, 5, 6], dtype=int64), Index(['joy', 'fear', 'anger', 'sadness', 'disgust', 'shame', 'guilt',
#        'disgust')


#       Use word embeddings. This is capable of capturing the context of a word in a sentence or document.
#       we get the actual texts from the data frame
#       Initialize the tokenizer with a 5000 word limit. This is the number of words we would like to encode.
#       we call fit_on_texts to create associations of words and numbers as shown in the image below.
text = isear_df.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(text)
# print(tokenizer.word_index)

#       calling text_to_sequence replaces the words in a sentence with their respective associated numbers.
#       This transforms each sentence into sequences of numbers.
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(text)
print(text[0])
print(encoded_docs[0])

#       The sentences or tweets have different number of words, therefore, the length of the sequence of numbers will be
#       different. Our model requires inputs to have equal lengths, so we will have to pad the sequence to have the
#       chosen length of inputs. This is done by calling the pad_sequence method with a length of 200.
#       All input sequences will have a length of 25.
padded_sequence = pad_sequences(encoded_docs, maxlen=MAXLEN_SEQ)
# print(padded_sequence[0])

embedding_vector_length = 32
# model = Sequential()
# model.add(Embedding(vocab_size, embedding_vector_length, input_length=MAXLEN_SEQ))
# model.add(SpatialDropout1D(0.25))
# model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
#
# history = model.fit(padded_sequence, emotion_label[0], validation_split=0.2, epochs=5, batch_size=32)
#
# test_word = "This is soo sad"
# tw = tokenizer.texts_to_sequences([test_word])
# tw = pad_sequences(tw, maxlen=MAXLEN_SEQ)
# prediction = int(model.predict(tw).round().item())
# print(emotion_label[1][prediction])

EMBEDDING_DIMENSION = 64
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIMENSION))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(EMBEDDING_DIMENSION, dropout=0.3, recurrent_dropout=0.3)))
model.add(Dense(EMBEDDING_DIMENSION, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(EMBEDDING_DIMENSION, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(1))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(padded_sequence, emotion_label[0], validation_split=0.2, epochs=10, batch_size=32)

test_word = "This is soo sad"
tw = tokenizer.texts_to_sequences([test_word])
tw = pad_sequences(tw, maxlen=MAXLEN_SEQ)
prediction = int(model.predict(tw).round().item())
print(emotion_label[1][prediction])
