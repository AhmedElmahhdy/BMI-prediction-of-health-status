from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load('https://drive.google.com/file/d/18D-Od0PWrpCvkavle-wT9fkXx0Sd-wO6/view?usp=sharing')

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
