import pickle
import flask
from flask import Flask, jsonify, request
import json
import numpy as np

app = Flask(__name__)

def load_models():
    file_name = "models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
        data_in = data['test']
    return model, data_in

@app.route('/predict', methods=['GET'])
def predict():
    # load model
    model, data_in = load_models()
    # stub input features
    x = np.array(data_in).reshape(1, -1)
    prediction = model.predict(x)[0]
    response = json.dumps({'response': prediction})
    return response, 200

if __name__ == "__main__":
    app.run(debug=True)