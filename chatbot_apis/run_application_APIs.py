from flask import Flask, request
from flask_restful import Resource, Api

from chatbot_apis.duck_duck_go.ddg_query import DuckDuckGoQuery
from chatbot_apis.facility.facility_finder import NearbyFacilityFinder
from chatbot_apis.mental_state.recommendation_based_on_mental_state import RecommendationBasedOnMentalState
from chatbot_apis.postcancer.post_cancer_practices import Practices

app = Flask(__name__)
api = Api(app)


class Prediction(Resource):
    def get(self):
        text = str(request.args.get('text'))
        recommendation_based_on_mental_status = RecommendationBasedOnMentalState()
        recommendation = recommendation_based_on_mental_status.get_recommendation(text)
        # return data and 200 OK code
        return recommendation, 200


class DuckDuckGo(Resource):
    def get(self):
        query = request.args.get('query')
        ddg_query = DuckDuckGoQuery()
        data = ddg_query.search(query)
        return data, 200


class FacilityFinder(Resource):
    def get(self):
        facility_type = request.args.get('facility_type')
        facility_finder = NearbyFacilityFinder()
        data = facility_finder.run(facility_type)
        return data, 200


class PostCancerPractices(Resource):
    def get(self):
        post_cancer_practices = Practices()
        data = post_cancer_practices.fetch()
        return data, 200


api.add_resource(Prediction, '/prediction')
api.add_resource(DuckDuckGo, '/search')
api.add_resource(FacilityFinder, '/facility')
api.add_resource(PostCancerPractices, '/practices')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
