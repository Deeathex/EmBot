from sentiment_analysis.nlp.NaturalLanguageProcessingModuleOld import NaturalLanguageProcessingModule
from sentiment_analysis.utils.Utils import Utils
from sentiment_analysis.utils.constants import DATA_FRAME_CSV_OUT_PATH, DATASET_CSV_PATH, DATA_FRAME_OUT_PATH_TEST, \
    DATASET_PATH_TEST, DATA_FRAME_PARQUET_OUT_PATH


class SentimentAnalysisScenariosWhenEmotionOccurs:
    def __init__(self, data_set_path, data_frame_out_path):
        self.__nlp = NaturalLanguageProcessingModule(data_set_path)
        self.__data_set_path = data_set_path
        self.__data_frame_out_path = data_frame_out_path

    def run_nlp(self):
        self.__nlp.preprocess_text(self.__data_frame_out_path)

    def run_nlp2(self):
        self.__nlp.preprocess_text2(self.__data_frame_out_path)

    def plot_data_distribution(self):
        data_frame = Utils.load_csv_file_as_data_frame(self.__data_set_path, separator='|')
        Utils.plot_data_distribution(data_frame)

    def run(self):
        pass


# sentiment_analysis_scenarios_when_emotion_occurs = SentimentAnalysisScenariosWhenEmotionOccurs(DATASET_CSV_PATH, DATA_FRAME_PARQUET_OUT_PATH)
# sentiment_analysis_scenarios_when_emotion_occurs.run_nlp()
sentiment_analysis_scenarios_when_emotion_occurs = SentimentAnalysisScenariosWhenEmotionOccurs(DATASET_PATH_TEST, DATA_FRAME_PARQUET_OUT_PATH)
sentiment_analysis_scenarios_when_emotion_occurs.run_nlp2()
# sentiment_analysis_scenarios_when_emotion_occurs.plot_data_distribution()
