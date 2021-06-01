from sentiment_analysis.application.mental_status_prediction import MentalStatusPredictor


class RecommendationBasedOnMentalState:
    def __init__(self):
        self.recommendations_by_mental_status = {
            'positive': ['no treatment needed'],
            'normal': ['no treatment needed'],
            'slightly stressed': [
                'You should think of some activities or hobbies you enjoy and practice them on a regular basis.'],
            'highly stressed': ['You should identify the stress sources from your life and to reduce them.'],
            'slightly depressed': ['Try to relax from time, to time. Try to meditate and free your mind.'],
            'highly depressed': [
                'When we are very sad for a longer period of time, we tend to alter the reality. Try to think about your emotions and feelings and try to analyse them.'],
        }

    def get_recommendation(self, text):
        mental_state = MentalStatusPredictor.run(text)
        data = {
            'mental_state': mental_state,
            'recommendation': self.recommendations_by_mental_status[mental_state]
        }
        return data
