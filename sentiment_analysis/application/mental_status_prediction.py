from sentiment_analysis.application.emotion_prediction import EmotionClassifier
from nltk.tokenize import sent_tokenize

from sentiment_analysis.utils.constants import PATH_TO_MODEL

KEEP_BALANCE_THRESHOLD = 15.0


class MentalStatusPredictor:
    def __init__(self, emotion_classifier):
        self.__emotion_classifier = emotion_classifier

    def get_mental_state(self, conversation):
        positive_emotions = 0
        negative_emotions = 0
        for text in conversation:
            emotion = self.__emotion_classifier.predict(text)
            if emotion == 'joy' or emotion == 'love':
                positive_emotions += 1
            else:
                negative_emotions += 1

        positivity_percent = (positive_emotions / len(conversation)) * 100
        negativity_percent = (negative_emotions / len(conversation)) * 100
        positivity_percent = positivity_percent + KEEP_BALANCE_THRESHOLD
        negativity_percent = negativity_percent - KEEP_BALANCE_THRESHOLD

        print('PP = ', positivity_percent)
        print('NP = ', negativity_percent)
        if positivity_percent >= 30 and negativity_percent < 20:
            return 'positive'
        if negativity_percent < 20:
            return 'normal'
        if 20 <= negativity_percent < 40:
            return 'slightly stressed'
        if 40 <= negativity_percent < 60:
            return 'highly stressed'
        if 60 <= negativity_percent < 80:
            return 'slightly depressed'
        if negativity_percent >= 80:
            return 'highly depressed'

    @staticmethod
    def run(text):
        mental_status_predictor = MentalStatusPredictor(EmotionClassifier(PATH_TO_MODEL))
        tokenized_text = sent_tokenize(text)
        return mental_status_predictor.get_mental_state(tokenized_text)


# mental_status_predictor = MentalStatusPredictor(EmotionClassifier(PATH_TO_MODEL))
# conversation1 = ['Today, honey, is a very beautiful day!', 'I wish to die soon..',
#                  "I want to die, I can't cope anymore!", 'I am so scared right now', 'All I want is some peace']
# print(mental_status_predictor.run(conversation1))
# conversation2 = ['Today, everything was awful!', 'I wish to die soon..',
#                  "I want to die, I can't cope anymore!", 'I am so scared right now', 'All I want is some peace']
# print(mental_status_predictor.run(conversation2))

# conversation3 = "I am not scared. I am getting better... Even if I am not, I try this. I thought life would be easier but it isn't"
# print(mental_status_predictor.run(conversation3))
