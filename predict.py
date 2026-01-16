from pathlib import Path

# Folder where your notebook lives
notebook_folder = r"C:\Users\Elitebook\Downloads\ml-100k"

# Full path for the predict.py file
predict_path = Path(notebook_folder) / "predict.py"

from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Load trained model
MODEL_PATH = 'best_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        best_model = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        f"{MODEL_PATH} not found. Make sure it exists in the same directory."
    )

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        'message': 'Random Forest Prediction API is running'
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if data is None or 'features' not in data:
        return jsonify({
            'error': 'Invalid input. Expected JSON with key "features".'
        }), 400

    try:
        features = np.array(data['features'], dtype=float).reshape(1, -1)
    except ValueError:
        return jsonify({
            'error': 'Features must be numeric.'
        }), 400

    expected_features = best_model.n_features_in_

    if features.shape[1] != expected_features:
        return jsonify({
            'error': f'Expected {expected_features} features, got {features.shape[1]}'
        }), 400

    prediction = int(best_model.predict(features)[0])
    probability = float(best_model.predict_proba(features)[0][1])

    return jsonify({
        'prediction': prediction,
        'probability': probability
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
