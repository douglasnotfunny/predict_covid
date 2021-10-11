import flask
import run
import requests
from flask import render_template

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/<location>', methods=['GET'])
def predict_world_tomorrow(location):
    return f"{str(run.get_predict_world(location)).split('.')[0]}"

app.run()
