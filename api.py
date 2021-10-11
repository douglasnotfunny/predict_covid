import flask
import run
import requests

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/<location>', methods=['GET'])
def predict_world_tomorrow(location):
    return f"{str(run.get_predict_world(location)).split('.')[0]}"

app.run()
