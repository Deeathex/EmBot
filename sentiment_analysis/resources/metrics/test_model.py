from sentiment_analysis.application.emotion_prediction import EmotionClassifier
from sentiment_analysis.utils.Utils import Utils
from sentiment_analysis.utils.constants import PATH_TO_MODEL, CARER_DATASET, EMOTION, SENTENCE


class StaticModelTesting:
    def __init__(self):
        self.__test_df = Utils.load_data_frame(path=CARER_DATASET + '/val.txt', separator='|')
        self.__emotion_classifier = EmotionClassifier(PATH_TO_MODEL)

    def run(self):
        test_entries_no = 0
        test_entries_correct = 0
        for i, row in self.__test_df.iterrows():
            test_entries_no += 1

            sentence = row[SENTENCE]
            actual_label = row[EMOTION]
            predicted_label = self.__emotion_classifier.predict(sentence)

            if actual_label == predicted_label:
                test_entries_correct += 1

        print('Correct: ', test_entries_correct)
        print('Total: ', test_entries_no)


static_model_testing = StaticModelTesting()
static_model_testing.run()
