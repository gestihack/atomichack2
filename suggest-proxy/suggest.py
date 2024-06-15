import os
import requests
from requests.auth import HTTPBasicAuth
from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/", methods=["POST"])
@cross_origin()
def proxy():
    
    req = {
        "suggest": {
            "question-suggest": {
            "prefix": request.get_json()["text"],
            "completion": {
                "field": "question",
                "fuzzy": {
                "fuzziness": "2"
                }
            }
            }
        }
    }

    x = requests.post(f"{os.environ["OPENSEARCH_URL"]}/question/_search", json=req, 
                      auth=HTTPBasicAuth(os.environ["OPENSEARCH_USER"], os.environ["OPENSEARCH_PASS"]), verify=False)

    return list(map(lambda sug: sug['text'], x.json()['suggest']['question-suggest'][0]["options"]))