import nltk

from sentence_transformers import SentenceTransformer
from sentiment_analysis.utils.Utils import Utils
from sentiment_analysis.utils.constants import DATASET_PATH, TEXT, TOKENIZED_SENTENCE, BERT_EMBEDDING


class NaturalLanguageProcessingModule:
    def __init__(self):
        nltk.download('punkt')
        self.__data_frame = Utils.load_csv_file_as_data_frame(DATASET_PATH, separator='|')

    def __apply_sentence_tokenization(self):
        self.__data_frame[TEXT] = self.__data_frame[TEXT].astype(str)
        self.__data_frame[TOKENIZED_SENTENCE] = self.__data_frame.apply(lambda row: nltk.sent_tokenize(row[TEXT]),
                                                                        axis=1)

    def __apply_bert_embedding(self):
        bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.__data_frame[BERT_EMBEDDING] = self.__data_frame.apply(
            lambda row: bert_model.encode(row[TOKENIZED_SENTENCE]),
            axis=1)

    def preprocess_text(self, output_data_frame):
        self.__apply_sentence_tokenization()
        self.__apply_bert_embedding()
        Utils.write_data_frame_to_csv(self.__data_frame, output_data_frame)
