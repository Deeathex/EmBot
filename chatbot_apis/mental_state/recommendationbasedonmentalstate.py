from chatbot_apis.mental_state.mental_status_prediction import MentalStatusPredictor

from random import randrange


class RecommendationBasedOnMentalState:
    def __init__(self):
        self.recommendations_by_mental_status = {
            'positive': ["I am glad you're feeling great!"],
            'normal': ["I am glad you're feeling great!"],
            'slightly stressed': [
                'You should think of some activities or hobbies you enjoy and practice them on a regular basis.',
                'You should meet up with your friends sometime and enjoy some mutual activities.',
                'I would suggest looking for a hobby group in your area.',
            ],
            'highly stressed': [
                'You should identify the stress sources from your life and reduce them.',
                'Try to think about what causes you the stress and try to avoid it.',
                'Try to discuss the stressful events in your life with someone close to you.',
            ],
            'slightly depressed': [
                'Try to relax from time, to time. Try to meditate and free your mind from the negative thoughts.',
                'You should try meditation sometime, it might help clear your mind.',
                'Everyone has bad days once in a while, but it is important not to generalize the negative thoughts however.',
            ],
            'highly depressed': [
                'When we are very sad for a longer period of time, we tend to alter the reality. Try to think about your emotions and feelings and try to analyse them. They are not good for you and your health.',
                'I know that everything seems to be falling apart, but it will get better if you focus on your positive thoughts, rather than the negative ones.',
                'There are moments when we think that our future is not in our hands, but those moments and the way we are coping with them matter. When in need, try to think about a pleasant memory that gives you the strength you need to overcome the challenges you are facing.',
            ],
        }

    def get_recommendation(self, text):
        mental_state = MentalStatusPredictor.run(text)
        data = {
            'mental_state': mental_state,
            'recommendation': self.__get_random_answer(self.recommendations_by_mental_status[mental_state])
        }
        return data

    def __get_random_answer(self, list):
        position = randrange(len(list))
        return list[position]
