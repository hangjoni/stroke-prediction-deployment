from flask import Flask, request, json, render_template
import pandas as pd
import pickle


app = Flask(__name__)

# dependencies
from utils import *


# Load your model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.get_json()

    query = pd.DataFrame(json_)
    preds = model.predict(query)
    probs = model.predict_proba(query)[:,1]
    
    df = pd.DataFrame({'pred': preds, 'prob': probs})

    return df.to_json()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
