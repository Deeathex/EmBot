import requests
import ipinfo
import googlemaps
from urllib.parse import quote

from flask import Flask, request
from flask_restful import Resource, Api

from chatbot_apis.mental_state.recommendationbasedonmentalstate import RecommendationBasedOnMentalState


app = Flask(__name__)
api = Api(app)


class Prediction(Resource):
    def get(self):
        text = request.args.get('text')
        print(text)
        text=str(text)
        recommendation_based_on_mental_status = RecommendationBasedOnMentalState()
        recommendation = recommendation_based_on_mental_status.get_recommendation(text)
        return recommendation, 200  # return data and 200 OK code


class DuckDuckGo(Resource):
    def get(self):
        query = request.args.get('query')
        url = "https://api.duckduckgo.com/?q=" + query + "&format=json"
        response = requests.get(url).json()
        print(response)

        text = response['AbstractText']
        image = response['Image']

        try:
            recommendation = response['RelatedTopics'][0]['FirstURL']
            if text is None or text == "":
                text = response['RelatedTopics'][0]['Text']
            if image is None or image == "":
                image = response['RelatedTopics'][0]['Icon']['URL']
        except:
            try:
                recommendation = response['RelatedTopics'][0]['Topics'][0]['FirstURL']
                if text is None or text == "":
                    text = response['RelatedTopics'][0]['Topics'][0]['Text']
                if image is None or image == "":
                    image = response['RelatedTopics'][0]['Topics'][0]['Icon']['URL']
            except:
                recommendation = ""
                text = ""
                image = ""

        if image is not None and image != "":
            image = "https://duckduckgo.com" + image

        data = {
            'text': text,
            'image': image,
            'recommendation': recommendation,
        }
        return data, 200


class FacilityFinder(Resource):
    __ipinfo_access_token = "6ca49ba6cf1a72"
    __ipinfo_handler = ipinfo.getHandler(__ipinfo_access_token)

    __gmaps_api_key = "AIzaSyAoaKxIatTMndm--8ED0ljb8XxzxP13sXg"
    __gmaps_client = googlemaps.Client(key=__gmaps_api_key)

    def get(self):
        facility_type = request.args.get('facility_type')
        if facility_type is None or facility_type == "":
            return {}, 401

        details = self.__ipinfo_handler.getDetails()
        # print(details.ip, " ", details.hostname, " ", details.city, " ", details.loc, " ", details.org, " ", details.postal)
        # 188.24.124.57   188-24-124-57.rdsnet.ro   Cluj-Napoca   46.7667,23.6000   AS8708 RCS & RDS SA   400001

        results = self.__gmaps_client.places_nearby(location=details.loc, rank_by="distance",
                                                    keyword=facility_type)

        if results["status"] == "OK" and len(results["results"]) > 0:
            results = results["results"]
            lat, long = results[0]["geometry"]["location"]["lat"], results[0]["geometry"]["location"]["lng"]
            url = "https://www.google.com/maps/search/?api=1&query=" + quote(str(lat) + "," + str(long)) + "&query_place_id=" + quote(results[0]["place_id"])
            message = "I searched for '" + facility_type + "' for you and found " + results[0]["name"] + " at address '" + results[0]["vicinity"] + "': " + url
        else:
            message = "I searched for '" + facility_type + "' for you, but found nothing relevant. Try something else instead?"

        data = {
            'text': message,
        }
        return data, 200


class PostCancerPractices(Resource):
    def get(self):
        cancer_type = request.args.get('cancer_type')
        # url = "https://api.duckduckgo.com/?q=" + query + "&format=json"
        # response = requests.get(url).json()

        data = {
        }
        return data, 200


api.add_resource(Prediction, '/prediction')
api.add_resource(DuckDuckGo, '/search')
api.add_resource(FacilityFinder, '/facility')
api.add_resource(PostCancerPractices, '/practices')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
