import pandas as pd
import requests
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Load your data
data = pd.read_csv('data.csv')

# Separate features and labels
y = data['stroke']
X = data.drop('stroke', axis=1)  # Assuming 'stroke' is the label column


# Select a few random data points for testing
X_test, _, y_test, _ = train_test_split(X, y, test_size=0.95, random_state=42)
print('number of test samples:', X_test.shape[0])

# Convert the test set to JSON
test_json = X_test.to_json(orient='records')

# URL of your Flask app
url = 'http://localhost:5001/predict'

# Send the POST request
headers = {'content-type': 'application/json'}
response = requests.post(url, data=test_json, headers=headers)

# Get the predictions
_predictions = response.json()
df = pd.DataFrame(_predictions)
preds = list(df.pred)
probs = list(df.prob)

# Calculate F1 score
f1 = f1_score(y_test, preds, average='binary')  # Adjust average as per your needs

print(f'F1 Score: {f1}')
