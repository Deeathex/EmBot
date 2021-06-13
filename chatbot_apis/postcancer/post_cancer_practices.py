import json

from random import randrange

PATH_JSON = 'D:\\Tot\\UBB\\SDI\\Semestrul_4\\Disertatie\\Parte practica\\Embot_nlp_backend\\chatbot_apis\\postcancer\\postcancer_practices.json'

POSTCANCER_PRACTICES = 'postcancer_practices'

DESCRIPTION = 'description'

RECOMMENDATION = 'recommendation'


class Practices:
    def __init__(self):
        self.__json_file = PATH_JSON
        pass

    def fetch(self):
        data_array = self.__load_practices()
        position = randrange(len(data_array))
        entry = data_array[position]

        data = {
            RECOMMENDATION: entry[RECOMMENDATION],
            DESCRIPTION: entry[DESCRIPTION],
        }
        return data

    def __load_practices(self):
        f = open(self.__json_file, encoding='utf-8')
        data = json.load(f)
        data_as_array = []
        for practice in data[POSTCANCER_PRACTICES]:
            data_as_array.append(practice)
        f.close()
        return data_as_array
