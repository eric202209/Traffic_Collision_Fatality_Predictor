from flask import Flask, request, jsonify, send_from_directory, render_template
import pickle
import numpy as np
import os

# app = Flask(__name__, static_folder='../static', static_url_path='')
app = Flask(__name__, static_folder='../static', static_url_path='/static')

# Load the model and scaler
model_path = 'models/best_model.pkl'
scaler_path = 'models/scaler.pkl'
feature_names_path = 'models/feature_names.pkl'

if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
else:
    print(f"Error: Model file is missing or empty. Path: {model_path}")
    model = None

if os.path.exists(scaler_path) and os.path.getsize(scaler_path) > 0:
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
else:
    print(f"Error: Scaler file is missing or empty. Path: {scaler_path}")
    scaler = None

if os.path.exists(feature_names_path) and os.path.getsize(feature_names_path) > 0:
    with open(feature_names_path, 'rb') as file:
        feature_names = pickle.load(file)
else:
    print(f"Error: Feature names file is missing or empty. Path: {feature_names_path}")
    feature_names = None

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Check if request.json contains expected keys
    expected_keys = set(feature_names)  # Assuming feature_names is a list of expected keys
    if not data or not expected_keys.issubset(data.keys()):
        return jsonify({'error': 'Missing or invalid features'}), 400

    # Ensure all expected features are present
    features = [data.get(feature, None) for feature in feature_names]

    # Handle missing or invalid features
    if None in features:
        return jsonify({'error': 'Missing or invalid features'}), 400

    # origin code
    if model is None or scaler is None or feature_names is None:
        return jsonify({'error': 'Model, scaler, or feature names not loaded properly'}), 500

    features = [data.get(feature, 0) for feature in feature_names]  # Use 0 as default if feature is missing
    features_array = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features_array)
    prediction = model.predict(scaled_features)
    return jsonify({'prediction': int(prediction[0])})

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)

if __name__ == '__main__':
    app.run(debug=True)