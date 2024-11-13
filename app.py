from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load the saved model and scaler
model = joblib.load('random_forest_model.pkl')


@app.route("/")
def welcome():
    return "Welcome to gym!"


@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the POST request
    data = request.get_json()
    print(data)
    # Extract features from the JSON
    features = pd.DataFrame([data['height'], data['weight'],  data['gender_male']],
                          columns=['Height', 'Weight',  'Gender_Male'])


    scaler = StandardScaler()

    features_scaled = scaler.transform(features)

    # Predict using the loaded model
    prediction = model.predict(features_scaled)

    print(prediction)
    
    # Return prediction as a JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(port=5000)