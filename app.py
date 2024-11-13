from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load('random_forest_model.pkl')

@app.route("/")
def home():
    return jsonify(message="Welcome to the Gym!")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Prepare features for prediction
    features = pd.DataFrame([[data['height'], data['weight'], data['gender_male']]],
                            columns=['Height', 'Weight', 'Gender_Male'])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Make a prediction
    prediction = model.predict(features_scaled)

    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(port=5000)
