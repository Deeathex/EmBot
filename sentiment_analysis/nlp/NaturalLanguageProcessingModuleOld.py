import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import wordnet

from sentence_transformers import SentenceTransformer
from sentiment_analysis.utils.Utils import Utils
from sentiment_analysis.utils.constants import TEXT, TOKENIZED_SENTENCE, BERT_EMBEDDING


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


class NaturalLanguageProcessingModule:
    def __init__(self, data_set_path):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        self.__data_frame = Utils.load_csv_file_as_data_frame(data_set_path, separator='|')
        self.__lemmatizer = WordNetLemmatizer()

    def __apply_sentence_tokenization(self):
        self.__data_frame[TEXT] = self.__data_frame[TEXT].astype(str)
        self.__data_frame[TOKENIZED_SENTENCE] = self.__data_frame.apply(lambda row: nltk.sent_tokenize(row[TEXT]),
                                                                        axis=1)

    def __apply_word_tokenization(self):
        self.__data_frame[TEXT] = self.__data_frame[TEXT].astype(str)
        self.__data_frame[TOKENIZED_SENTENCE] = self.__data_frame[TEXT].apply(nltk.word_tokenize)

    def __encode_and_flatten(self, bert_model, tokenized_sentences):
        encoded_sentences = bert_model.encode(tokenized_sentences, convert_to_numpy=True, normalize_embeddings=True)
        flat_list = []
        for sublist in encoded_sentences:
            for item in sublist:
                flat_list.append(item)
        return np.asarray(flat_list)

    def __apply_bert_embedding(self):
        bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.__data_frame[BERT_EMBEDDING] = self.__data_frame.apply(
            lambda row: self.__encode_and_flatten(bert_model, row[TOKENIZED_SENTENCE]),
            axis=1)

    def preprocess_text(self, output_data_frame):
        self.__apply_sentence_tokenization()
        self.__apply_bert_embedding()
        Utils.write_data_frame_to_parquet(self.__data_frame, output_data_frame)

    def stem_list(self, dataframe, stemmer):
        my_list = self.__data_frame[TEXT]
        stemmed_list = [stemmer.stem(word) for word in my_list]
        return (stemmed_list)

    def lemmatize_text(self, text, lemmatizer, w_tokenizer):
        return [lemmatizer.lemmatize(w, 'v') for w in w_tokenizer.tokenize(text)]

    def preprocess_text2(self, output_data_frame):
        # lowercase
        self.__data_frame[TEXT] = self.__data_frame[TEXT].apply(lambda x: " ".join(x.lower() for x in x.split()))
        #  remove punctuation
        self.__data_frame[TEXT] = self.__data_frame[TEXT].str.replace('[^\w\s]', '')
        # remove stopwords
        stop = stopwords.words('english')
        self.__data_frame[TEXT] = self.__data_frame[TEXT].apply(
            lambda x: " ".join(x for x in x.split() if x not in stop))
        # print(self.__data_frame[TEXT])
        # lemmatization
        self.__data_frame[TEXT] = self.__data_frame[TEXT].apply(lambda x: self.lemmatize_sentence(x))
        print(self.__data_frame[TEXT])
        # tokenization
        self.__apply_word_tokenization()
        print(self.__data_frame[TOKENIZED_SENTENCE])
        # stemming
        # stemmer = PorterStemmer()
        # self.__data_frame[TOKENIZED_SENTENCE] = self.__data_frame[TOKENIZED_SENTENCE].apply(lambda x: self.stem_list)
        # lemmatization
        # w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        # lemmatizer = nltk.stem.WordNetLemmatizer()
        # self.__data_frame[TOKENIZED_SENTENCE] = self.__data_frame[TOKENIZED_SENTENCE].apply(lambda x: self.lemmatize_text(x, lemmatizer, w_tokenizer))
        # lemmatizer = WordNetLemmatizer()
        # self.__data_frame[TEXT] = self.__data_frame[TEXT].apply(lambda x: self.lemmatize_text(x[TOKENIZED_SENTENCE], lemmatizer))

        self.__apply_bert_embedding()
        print(self.__data_frame[BERT_EMBEDDING])
        Utils.write_data_frame_to_parquet(self.__data_frame, output_data_frame)

    def lemmatize_sentence(self, sentence):
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
