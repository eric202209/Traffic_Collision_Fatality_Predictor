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

def load_pickle(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'rb') as file:
            return pickle.load(file)
    else:
        print(f"Error: File is missing or empty. Path: {path}")
        return None

model = load_pickle(model_path)
scaler = load_pickle(scaler_path)
feature_names = load_pickle(feature_names_path)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse incoming JSON data
        data = request.get_json(force=True)
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid data format. Expected a JSON object'}), 400
        
        if model is None or scaler is None or feature_names is None:
            return jsonify({'error': 'Model, scaler, or feature names not loaded properly'}), 500
        
        # Ensure all expected features are present
        missing_features = [feature for feature in feature_names if feature not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {", ".join(missing_features)}'}), 400
        
        # Extract features in the correct order
        features = [float(data[feature]) for feature in feature_names]
        
        features_array = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)
        prediction = model.predict(scaled_features)
        
        return jsonify({'prediction': int(prediction[0])})
    
    except ValueError as ve:
        return jsonify({'error': f'Invalid value for a feature: {str(ve)}'}), 400
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred during prediction'}), 500


@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('../static/images', filename)

if __name__ == '__main__':
    app.run(debug=True)