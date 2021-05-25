from sentiment_analysis.application.emotion_prediction import EmotionClassifier
from nltk.tokenize import sent_tokenize


class MentalStatusPredictor:
    def __init__(self, emotion_classifier):
        self.__emotion_classifier = emotion_classifier

    def run(self, conversation):
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


mental_status_predictor = MentalStatusPredictor(EmotionClassifier('../metrics/saved_model_2021-05-07_14-28-54.h5'))
# conversation1 = ['Today, honey, is a very beautiful day!', 'I wish to die soon..',
#                  "I want to die, I can't cope anymore!", 'I am so scared right now', 'All I want is some peace']
# print(mental_status_predictor.run(conversation1))
# conversation2 = ['Today, everything was awful!', 'I wish to die soon..',
#                  "I want to die, I can't cope anymore!", 'I am so scared right now', 'All I want is some peace']
# print(mental_status_predictor.run(conversation2))

conversation3 = "I am not scared. I am getting better... Even if I am not, I try this. I thought life would be easier but it isn't"
tokenized_sentence = sent_tokenize(conversation3)
print(tokenized_sentence)
print(mental_status_predictor.run(tokenized_sentence))