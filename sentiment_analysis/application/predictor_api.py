from flask import Flask, request
from flask_restful import Resource, Api

from sentiment_analysis.application.recommendationbasedonmentalstate import RecommendationBasedOnMentalState

app = Flask(__name__)
api = Api(app)


class Predictor(Resource):
    def get(self):
        text = request.args.get('text')
        recommendation_based_on_mental_status = RecommendationBasedOnMentalState()
        recommendation = recommendation_based_on_mental_status.get_recommendation(text)
        return recommendation, 200  # return data and 200 OK code


api.add_resource(Predictor, '/predictor')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
