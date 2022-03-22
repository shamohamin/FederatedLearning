from flask import Flask, request, make_response
import json
import pickle

app = Flask(__name__)

@app.route("/get_weights", methods=["POST"])
def get_model():
    data = json.loads(request.data)
    assert type(data) is dict
    if "weights" not in data.keys():
        return make_response({"message": "weights are not provided"}, 400)

    encoded_weights = data["weights"]
    weights = pickle.loads(encoded_weights)
    print(weights)
    
@app.route("/index", methods=["GET"])
def index():
    return "<h1> HELLO </h1>"

app.run()