import json

from random import randrange

from chatbot_apis.mental_state.mental_status_prediction import MentalStatusPredictor

RECOMMENDATIONS_BY_MENTAL_STATUS = 'recommendations_by_mental_status'

RECOMMENDATION = 'recommendation'
MENTAL_STATE = 'mental_state'

PATH_JSON = 'D:\\Tot\\UBB\\SDI\\Semestrul_4\\Disertatie\\Parte practica\\Embot_nlp_backend\\chatbot_apis\\mental_state\\recommendations_by_mental_state.json'


class RecommendationBasedOnMentalState:
    def __init__(self):
        self.__json_file = PATH_JSON
        self.__recommendations_by_mental_status = self.__load_practices()

    def get_recommendation(self, text):
        mental_state = MentalStatusPredictor.run(text)
        recommendations = []
        for pair in self.__recommendations_by_mental_status[RECOMMENDATIONS_BY_MENTAL_STATUS]:
            if pair[MENTAL_STATE] == mental_state:
                recommendations = pair['recommendation']
        data = {
            MENTAL_STATE: mental_state,
            RECOMMENDATION: self.__get_random_answer(recommendations)
        }
        return data

    def __load_practices(self):
        f = open(self.__json_file, encoding='utf-8')
        data = json.load(f)
        f.close()
        return data

    @staticmethod
    def __get_random_answer(list):
        position = randrange(len(list))
        return list[position]
