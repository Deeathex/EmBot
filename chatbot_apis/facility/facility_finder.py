import ipinfo
import googlemaps
from urllib.parse import quote

VICINITY = "vicinity"

NAME = "name"
PLACE_ID = "place_id"
LONGITUDE = "lng"
LATITUDE = "lat"
LOCATION = "location"
GEOMETRY = "geometry"
RESULTS = "results"

DATA_TEXT = 'text'


class NearbyFacilityFinder:
    def __init__(self):
        self.__ipinfo_access_token = "6ca49ba6cf1a72"
        self.__ipinfo_handler = ipinfo.getHandler(self.__ipinfo_access_token)

        self.__gmaps_api_key = "AIzaSyAoaKxIatTMndm--8ED0ljb8XxzxP13sXg"
        self.__gmaps_client = googlemaps.Client(key=self.__gmaps_api_key)

    def run(self, facility_type):
        if facility_type is None or facility_type == "":
            return {}, 401

        details = self.__ipinfo_handler.getDetails()
        # print(details.ip, " ", details.hostname, " ", details.city, " ", details.loc, " ", details.org, " ", details.postal)
        # 188.24.124.57   188-24-124-57.rdsnet.ro   Cluj-Napoca   46.7667,23.6000   AS8708 RCS & RDS SA   400001

        results = self.__gmaps_client.places_nearby(location=details.loc, rank_by="distance",
                                                    keyword=facility_type)

        if results["status"] == "OK" and len(results[RESULTS]) > 0:
            results = results[RESULTS]
            lat, long = results[0][GEOMETRY][LOCATION][LATITUDE], results[0][GEOMETRY][LOCATION][LONGITUDE]
            url = "https://www.google.com/maps/search/?api=1&query=" + quote(
                str(lat) + "," + str(long)) + "&query_place_id=" + quote(results[0][PLACE_ID])
            message = "I searched for '" + facility_type + "' for you and found " + results[0][
                NAME] + " at address '" + results[0][VICINITY] + "': " + url
        else:
            message = "I searched for '" + facility_type + "' for you, but found nothing relevant. Try something else instead?"

        data = {
            DATA_TEXT: message,
        }

        return data
