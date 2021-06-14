# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
import requests

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

from random import randrange

DESCRIPTION = "description"
RECOMMENDATION = "recommendation"
IMAGE = "image"
TEXT = "text"
QUERY = "query"
FACILITY_TYPE = "facility_type"
EMPTY_STRING = ""

cbt_messages = []


class ActionAddFeeling(Action):

    def name(self) -> Text: return "action_add_feeling_message"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global cbt_messages
        message = str(tracker.latest_message[TEXT])
        cbt_messages.append(message)
        return []


class ActionCheckMentalStateFromMessages(Action):

    def name(self) -> Text:
        return "action_check_mental_state_from_messages"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global cbt_messages
        responses = ['About what?', 'First, tell me something about you.',
                     'To have an opinion, you first have to provide me some information about the context.']
        if not cbt_messages:
            position = randrange(len(responses))
            dispatcher.utter_message(responses[position])
            return []
        text = EMPTY_STRING
        for message in cbt_messages:
            text += message + '. '
        url = "http://127.0.0.1:5000/prediction?text=" + text
        response = requests.get(url).json()

        cbt_messages = []
        dispatcher.utter_message(str(response[RECOMMENDATION]))
        return []


class ActionFaciltySearch(Action):

    def name(self) -> Text: return "action_facility_search"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        facility = tracker.get_slot(FACILITY_TYPE)
        url = "http://127.0.0.1:5000/facility?facility_type=" + facility
        response = requests.get(url).json()
        SlotSet(FACILITY_TYPE, EMPTY_STRING)

        dispatcher.utter_message(str(response[TEXT]))
        return []


class ActionSearchKeyword(Action):

    def name(self) -> Text:
        return "action_search_keyword"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        query = tracker.get_slot(QUERY)
        if query is None:
            return []

        url = "http://127.0.0.1:5000/search?query=" + query
        response = requests.get(url).json()
        SlotSet(QUERY, EMPTY_STRING)

        text = response[TEXT]
        image_link = response[IMAGE]
        link = response[RECOMMENDATION]

        if text is None or text == EMPTY_STRING:
            dispatcher.utter_message("Sorry, I couldn't find anything. Search for something else instead?")
            return []

        dispatcher.utter_message(text)
        if image_link is not None and image_link != EMPTY_STRING:
            dispatcher.utter_message("![testText](" + image_link + ")")
        if link is not None and link != EMPTY_STRING:
            dispatcher.utter_message("Here's a link if you want to know more: " + str(link))

        return []


class ActionPostCancerPractices(Action):

    def name(self) -> Text:
        return "action_post_cancer_practices"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        url = "http://127.0.0.1:5000/practices"
        response = requests.get(url).json()

        recommendation = response[RECOMMENDATION]
        description = response[DESCRIPTION]

        dispatcher.utter_message(recommendation + ': ' + description)
        return []
