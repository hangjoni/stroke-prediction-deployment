from flask import Flask, request, render_template, url_for, redirect, session
import pandas as pd
import pickle


app = Flask(__name__)
app.secret_key= 'secret'

# dependencies
from utils import *


# Load your model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# end point to serve front end prediction with one input
@app.route('/', methods=['GET', 'POST'])
def predict_one_datapoint():
    if request.method=='POST':
        data = {
            'gender': int(request.form.get('gender')),
            'age': int(request.form.get('age')),
            'hypertension': int(request.form.get('hypertension')),
            'heart_disease': int(request.form.get('heart_disease')),
            'work_type': int(request.form.get('work_type')),
            'avg_glucose_level': float(request.form.get('avg_glucose_level')),
            'bmi': float(request.form.get('bmi'))
        }
        # data = request.form.to_dict()
        preds, probs = _predict_helper(data)
        pred, prob = preds[0], probs[0]

        data['pred'] = str(pred)
        data['prob'] = str(prob)
        session['form_data'] = data

        return redirect(url_for('predict_one_datapoint'))
    
    form_data = session.get('form_data', {})
    return render_template('index.html', form_data=form_data)

# end point for batch prediction
@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()

    preds, probs = _predict_helper(json_data)
    
    df = pd.DataFrame({'pred': preds, 'prob': probs})

    return df.to_json()

def _predict_helper(json_data):
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
