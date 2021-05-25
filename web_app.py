from flask import Flask
from flask import render_template, jsonify, request
import requests
from urllib.request import urlopen
import json
from models import *

import random

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


app.config["DEBUG"] = True
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000)
