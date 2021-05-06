import nltk
import numpy as np
import re

import gensim.downloader as api

from keras.preprocessing.text import Tokenizer

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sentiment_analysis.utils.Utils import Utils
from sentiment_analysis.utils.constants import SENTENCE, PROCESSED_SENTENCE, EMOTION


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return 'a'
    elif nltk_tag.startswith('V'):
        return 'v'
    elif nltk_tag.startswith('N'):
        return 'n'
    elif nltk_tag.startswith('R'):
        return 'r'
    else:
        return None


class NLPModule:
    def __init__(self, num_words, path_to_dataset):
        self.__train_dataframe = Utils.load_data_frame(path=path_to_dataset + '/train.txt', separator='|')
        self.__validation_dataframe = Utils.load_data_frame(path=path_to_dataset + '/val.txt', separator='|')
        self.__num_words = num_words
        self.__tokenizer = Tokenizer(num_words, lower=True)
        self.__lemmatizer = WordNetLemmatizer()
        self.__labeled_dict = {'joy': 0, 'anger': 1, 'love': 2, 'sadness': 3, 'fear': 4, 'surprise': 5}
        self.run_nlp(self.__train_dataframe)
        self.run_nlp(self.__validation_dataframe)

    def get_vocabulary_size(self):
        return self.__num_words

    def get_train_dataframe(self):
        return self.__train_dataframe

    def get_validation_dataframe(self):
        return self.__validation_dataframe

    def get_tokenizer(self):
        return self.__tokenizer

    def run_nlp(self, dataframe):
        print('Encoding output values..')
        dataframe[EMOTION] = dataframe.emotion.replace(self.__labeled_dict)

        print('Lowercasing sentences..')
        dataframe[PROCESSED_SENTENCE] = dataframe[SENTENCE].apply(
            lambda x: " ".join(x.lower() for x in x.split()))

        print('Removing punctuation from sentences..')
        dataframe[PROCESSED_SENTENCE] = dataframe[PROCESSED_SENTENCE].str.replace('[^\w\s]', '',
                                                                                  regex=True)

        # print('Removing stopwords from sentences...')
        # stop = stopwords.words('english')
        # self.train_dataframe[PROCESSED_SENTENCE] = self.train_dataframe[PROCESSED_SENTENCE].apply(
        #     lambda x: " ".join(x for x in x.split() if x not in stop))

        print('Expanding contractions from sentences..')
        dataframe[PROCESSED_SENTENCE] = dataframe[PROCESSED_SENTENCE].apply(lambda x: self.__decontracted(x))

        print('Performing lemmatization on sentences..')
        dataframe[PROCESSED_SENTENCE] = dataframe[PROCESSED_SENTENCE].apply(lambda x: self.__lemmatize_sentence(x))

        print('Performing tokenization on sentences..')
        self.__tokenizer.fit_on_texts(dataframe.sentence.values)

    def compute_word2vec_weight_matrix(self):
        # creation of word2vec weight matrix
        glove_gensim = api.load('glove-wiki-gigaword-100')  # 100 dimension
        vector_size = 100
        gensim_weight_matrix = np.zeros((self.__num_words, vector_size))

        for word, index in self.__tokenizer.word_index.items():
            if index < self.__num_words:  # since index starts with zero
                if word in glove_gensim.wv.vocab:
                    gensim_weight_matrix[index] = glove_gensim[word]
                else:
                    gensim_weight_matrix[index] = np.zeros(100)

        return gensim_weight_matrix

    def __lemmatize_sentence(self, sentence):
        # tokenize the sentence and find the POS tag for each token
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
        # tuple of (token, wordnet_tag)
        wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                # if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:
                # else use the tag to lemmatize the token
                lemmatized_sentence.append(self.__lemmatizer.lemmatize(word, tag))
        return " ".join(lemmatized_sentence)

    def __decontracted(self, phrase):
        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"ive", "i have", phrase)
        phrase = re.sub(r"didnt", "did not", phrase)
        phrase = re.sub(r"im", "i am", phrase)
        phrase = re.sub(r"could've", "could have", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase
