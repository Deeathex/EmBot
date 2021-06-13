import requests

IMAGE = 'Image'
ABSTRACT_TEXT = 'AbstractText'
EMPTY_STRING = ""
TOPICS = 'Topics'
URL = 'URL'
ICON = 'Icon'
TEXT = 'Text'
FIRST_URL = 'FirstURL'
RELATED_TOPICS = 'RelatedTopics'

DATA_TEXT = 'text'
DATA_IMAGE = 'image'
DATA_RECOMMENDATION = 'recommendation'


class DuckDuckGoQuery:
    def __init__(self):
        self.__base_url = 'https://api.duckduckgo.com/'
        self.__image_url = 'https://duckduckgo.com'

    def search(self, query):

        url = self.__base_url + '?q=' + query + '&format=json'
        response = requests.get(url).json()
        print(response)

        text = response[ABSTRACT_TEXT]
        image = response[IMAGE]

        try:
            recommendation = response[RELATED_TOPICS][0][FIRST_URL]
            if text is None or text == EMPTY_STRING:
                text = response[RELATED_TOPICS][0][TEXT]
            if image is None or image == EMPTY_STRING:
                image = response[RELATED_TOPICS][0][ICON][URL]
        except:
            try:
                recommendation = response[RELATED_TOPICS][0][TOPICS][0][FIRST_URL]
                if text is None or text == EMPTY_STRING:
                    text = response[RELATED_TOPICS][0][TOPICS][0][TEXT]
                if image is None or image == EMPTY_STRING:
                    image = response[RELATED_TOPICS][0][TOPICS][0][ICON][URL]
            except:
                recommendation = EMPTY_STRING
                text = EMPTY_STRING
                image = EMPTY_STRING

        if image is not None and image != EMPTY_STRING:
            image = self.__image_url + image

        data = {
            DATA_TEXT: text,
            DATA_IMAGE: image,
            DATA_RECOMMENDATION: recommendation,
        }

        return data
