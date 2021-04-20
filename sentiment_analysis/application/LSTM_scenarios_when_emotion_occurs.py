from sentiment_analysis.nlp.NaturalLanguageProcessingModule import NaturalLanguageProcessingModule
from sentiment_analysis.utils.Utils import Utils
from sentiment_analysis.utils.constants import DATA_FRAME_PATH, DATASET_PATH


class SentimentAnalysisScenariosWhenEmotionOccurs:
    def __init__(self):
        self.__nlp = NaturalLanguageProcessingModule()

    def run_nlp(self):
        self.__nlp.preprocess_text(DATA_FRAME_PATH)

    def plot_data_distribution(self):
        data_frame = Utils.load_csv_file_as_data_frame(DATASET_PATH, separator='|')
        Utils.plot_data_distribution(data_frame)

    def run(self):
        pass


sentiment_analysis_scenarios_when_emotion_occurs = SentimentAnalysisScenariosWhenEmotionOccurs()
# sentiment_analysis_scenarios_when_emotion_occurs.run_nlp()
sentiment_analysis_scenarios_when_emotion_occurs.plot_data_distribution()
