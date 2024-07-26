from flask import Flask, request, jsonify, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from utils.logger import setup_logger
import pickle
import numpy as np
import os
import pandas as pd

app = Flask(__name__, static_folder='../static', static_url_path='/static')
limiter = Limiter(key_func=get_remote_address)
limiter.init_app(app)
logger = setup_logger()

def load_pickle(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'rb') as file:
            return pickle.load(file)
    else:
        app.logger.error(f"Error: File is missing or empty. Path: {path}")
        return None
    
# Load the model, scaler, metrics, SHAP values, and feature names
model = load_pickle('models/best_model.pkl')
scaler = load_pickle('models/scaler.pkl')
metrics = load_pickle('models/best_model_metrics.pkl')
shap_values = load_pickle('models/best_model_shap_values.pkl')
feature_names = load_pickle('models/feature_names.pkl')

# Load the original dataset for visualization
df = pd.read_csv('Killed_and_Seriously_Injured.csv')

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
@limiter.limit("5 per minute")
def predict():
    try:
        # Parse incoming JSON data
        data = request.get_json(force=True)
        app.logger.info(f"Received data: {data}")

        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid data format'}), 400
        
        if model is None or scaler is None or feature_names is None:
            return jsonify({'error': 'Model components not loaded properly'}), 500
                
        # Extract and validate features
        features = []
        for feature in feature_names:
            value = data.get(feature)
            if value is None:
                app.logger.error(f"Missing value for feature: {feature}")
                return jsonify({'error': f'Missing value for feature: {feature}'}), 400
            try:
                features.append(float(value))
            except ValueError:
                return jsonify({'error': f'Invalid value for feature {feature}: {value}'}), 400

        features_array = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]  # Probability of fatal outcome
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability)
        })
        
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_fatalities')
def get_fatalities():
    fatalities = df[df['ACCLASS'] == 'Fatal']
    return jsonify(fatalities[['LATITUDE', 'LONGITUDE']].to_dict('records'))

@app.route('/model_metrics')
def model_metrics():
    return jsonify(metrics)

@app.route('/feature_importance')
def feature_importance():
    with open('models/best_model_shap_values.pkl', 'rb') as f:
        shap_values = pickle.load(f)
    
    feature_imp = np.abs(shap_values).mean(0)
    feature_imp = feature_imp.flatten()  # If it's a 2D array
    # feature_imp = feature_imp[:, 0]  # If you want to select the first column
    return jsonify([{'feature': f, 'importance': float(i)} for f, i in zip(feature_names, feature_imp)])
    # return jsonify([{'feature': f, 'importance': float(i.item())} for f, i in zip(feature_names, feature_imp)])

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('../static/images', filename)

if __name__ == '__main__':
    app.run(debug=True)