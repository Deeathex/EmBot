import numpy as np

from keras.models import load_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nltk.stem.wordnet import WordNetLemmatizer

from sentiment_analysis.nlp.NLPModule import NLPModule
from sentiment_analysis.utils.Utils import Utils
from sentiment_analysis.utils.constants import NUM_WORDS_VOCABULARY, MAXLEN, CARER_DATASET


class EmotionClassifier:
    def __init__(self, path_to_model):
        self.__classifier = load_model(path_to_model)
        self.__labeled_dict = {'joy': 0, 'anger': 1, 'love': 2, 'sadness': 3, 'fear': 4, 'surprise': 5}
        self.__tokenizer = Tokenizer(NUM_WORDS_VOCABULARY, lower=True)
        self.__train_df = Utils.load_data_frame(path=CARER_DATASET + '/train.txt', separator='|')
        self.__tokenizer.fit_on_texts(self.__train_df.sentence.values)

    def __get_key(self, value):
        for key, val in self.__labeled_dict.items():
            if (val == value):
                return key

    def predict(self, sentence):
        sentence = sentence.lower()
        sentence = NLPModule.remove_punctuation(sentence)
        sentence = NLPModule.decontracted(sentence)
        NLPModule.lemmatize_sentence(sentence, WordNetLemmatizer())

        sentence_lst = []
        sentence_lst.append(sentence)
        sentence_seq = self.__tokenizer.texts_to_sequences(sentence_lst)
        sentence_padded = pad_sequences(sentence_seq, maxlen=MAXLEN, padding='post')
        predicted_value = self.__classifier.predict_classes(sentence_padded)
        ans = self.__get_key(predicted_value[0])
        print('For sentence: ', sentence)
        print('The emotion predicted is', ans)
        return ans


emotion_classifier = EmotionClassifier('../metrics/saved_model_2021-05-07_14-28-54.h5')
emotion_classifier.predict('What a wonderful day!')
emotion_classifier.predict('Today is a very beautiful day!')
emotion_classifier.predict('I wish I was dead')
emotion_classifier.predict('I want to die')
emotion_classifier.predict('I am so stressed')
emotion_classifier.predict('I want to cry')
emotion_classifier.predict('Im so pathetic')
emotion_classifier.predict('Im defeated by this situation')
