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


class ActionScenariosWhenEmotionOccurs(Action):

    def name(self) -> Text: return "action_scenarios_when_emotion_occurs"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # get the latest message
        # print(tracker.latest_message)
        # for blob in tracker.latest_message['entities']:
        #     print(tracker.latest_message)
        #     print(blob['entity'])
        #     print(blob['value'])
        #     print(blob)
        #     print('________')
        # dispatcher.utter_message(text="Hello World!")
        # print(str(tracker.latest_message))
        text = str(tracker.latest_message['text'])
        url = "http://127.0.0.1:5000/predictor?text=" + text
        print(url)
        response = requests.get(url)
        print(response.status_code)
        dispatcher.utter_message(str(response.json()))
        return []


class ActionFaciltySearch(Action):

    def name(self) -> Text: return "action_facility_search"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        facility = tracker.get_slot("facility_type")
        address = "Galautas, Centru"
        dispatcher.utter_message(text="Here is the address of the {}:{}".format(facility, address))
        return []
