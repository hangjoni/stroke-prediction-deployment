from flask import Flask, request, json, render_template
import pandas as pd
import pickle


app = Flask(__name__)

# dependencies
from utils import *


# Load your model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# end point to serve front end prediction with one input
@app.route('/', methods=['GET', 'POST'])
def predict_one_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data = {
            'gender': int(request.form.get('gender')),
            'age': int(request.form.get('age')),
            'hypertension': int(request.form.get('hypertension')),
            'heart_disease': int(request.form.get('heart_disease')),
            'work_type': int(request.form.get('work_type')),
            'avg_glucose_level': float(request.form.get('avg_glucose_level')),
            'bmi': float(request.form.get('bmi'))
        }
        preds, probs = _predict_helper(data)
        pred, prob = preds[0], probs[0]

        return render_template('index.html', pred=pred, prob=prob)

# end point for batch prediction
@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()

    preds, probs = _predict_helper(json_data)
    
    df = pd.DataFrame({'pred': preds, 'prob': probs})

    return df.to_json()

def _predict_helper(json_data):
    print('json_data')
    print(json_data)
    # json_data could be dictionary or list of dictionary
    if isinstance(json_data, list):
        query = pd.DataFrame(json_data)
    else:
        query = pd.DataFrame([json_data])
    preds = model.predict(query)
    probs = model.predict_proba(query)[:,1]
    return preds, probs


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
