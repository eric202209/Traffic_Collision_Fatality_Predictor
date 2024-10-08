from flask import Flask, request, jsonify, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from utils.logger import setup_logger
import os
import pickle
import pandas as pd
import numpy as np
import shap
import json
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils

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

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
@limiter.limit("5 per minute")
def predict():
    try:
        data = request.get_json(force=True)
        app.logger.info(f"Received data: {data}")

        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid data format'}), 400
        
        if model is None or scaler is None or feature_names is None:
            return jsonify({'error': 'Model components not loaded properly'}), 500
                
        # Create a dictionary with all features set to 0
        features = {feature: 0 for feature in feature_names}
        
        # Update the features with the received data
        for key, value in data.items():
            if key in features:
                features[key] = float(value)

        # Convert the features dictionary to a list in the correct order
        features_array = [features[feature] for feature in feature_names]
        features_array = np.array(features_array).reshape(1, -1)

        app.logger.info(f"Processed features: {features}")

        scaled_features = scaler.transform(features_array)
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]

        # Generate SHAP values for this prediction
        explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.KernelExplainer(model.predict_proba, scaled_features)
        shap_values = explainer.shap_values(scaled_features)
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'shap_values': shap_values[0].tolist()  # Assuming binary classification
        })
        
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_fatalities')
def get_fatalities():
    df = pd.read_csv('Killed_and_Seriously_Injured.csv')
    fatalities = df[df['ACCLASS'] == 'Fatal']
    return jsonify(fatalities[['LATITUDE', 'LONGITUDE']].to_dict('records'))

@app.route('/model_metrics')
def model_metrics():
    return jsonify(metrics)

@app.route('/feature_importance')
def feature_importance():
    feature_imp = np.abs(shap_values).mean(0)
    feature_imp = feature_imp.flatten()
    return jsonify([{'feature': f, 'importance': float(i)} for f, i in zip(feature_names, feature_imp)])

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('../static/images', filename)

@app.route('/time_analysis')
def time_analysis():
    df = pd.read_csv('Killed_and_Seriously_Injured.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['Hour'] = df['DATE'].dt.hour
    df['DayOfWeek'] = df['DATE'].dt.dayofweek
    df['Month'] = df['DATE'].dt.month

    hourly_counts = df.groupby('Hour')['ACCLASS'].count().reset_index()
    daily_counts = df.groupby('DayOfWeek')['ACCLASS'].count().reset_index()
    monthly_counts = df.groupby('Month')['ACCLASS'].count().reset_index()

    return jsonify({
        'hourly': hourly_counts.to_dict('records'),
        'daily': daily_counts.to_dict('records'),
        'monthly': monthly_counts.to_dict('records')
    })

@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    data = request.get_json(force=True)
    predictions = []
    for scenario in data:
        # Create a dictionary with all features set to 0
        features = {feature: 0 for feature in feature_names}
        
        # Update the features with the received data
        for key, value in scenario.items():
            if key in features:
                features[key] = float(value)

        app.logger.info(f"Received scenario: {scenario}")

        # Convert the features dictionary to a list in the correct order
        features_array = [features[feature] for feature in feature_names]
        features_array = np.array(features_array).reshape(1, -1)
        app.logger.info(f"Processed features: {features}")

        scaled_features = scaler.transform(features_array)
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]
        predictions.append({
            'prediction': int(prediction),
            'probability': float(probability)
        })
        app.logger.info(f"Prediction: {prediction}, Probability: {probability}")
    return jsonify(predictions)

@app.route('/shap_summary')
def shap_summary():
    try:
        # Load or calculate SHAP values
        shap_values = load_pickle('models/best_model_shap_values.pkl')
                
        if shap_values is None:
            return jsonify({'error': 'SHAP values not found'}), 404

        # Assume shap_values is a 3D array (samples, features, classes)
        if len(shap_values.shape) == 3:
            # Aggregate SHAP values: mean absolute value across samples
            shap_values = np.abs(shap_values).mean(axis=0)
        
        # Log shapes for debugging
        logger.info(f"SHAP values shape after aggregation: {shap_values.shape}")

        # Ensure shap_values is 2D (features, classes)
        if len(shap_values.shape) != 2:
            raise ValueError("SHAP values are not in the expected 2D format.")
        
        # Calculate mean importance across classes if needed
        feature_importance = shap_values.mean(axis=1)
        
        # Create a DataFrame for easier plotting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        })

        # Check DataFrame shape
        logger.info(f"Importance DataFrame shape: {importance_df.shape}")
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Create the plot data
        plot_data = {
            'data': [
                {
                    'type': 'bar',
                    'x': importance_df['feature'].tolist(),
                    'y': importance_df['importance'].tolist(),
                    'name': 'Feature Importance'
                }
            ],
            'layout': {
                'title': 'Feature Importance (SHAP)',
                'xaxis': {'title': 'Feature'},
                'yaxis': {'title': 'Importance'},
                'barmode': 'group'
            }
        }
        logger.info(f"Plot data: {plot_data}")        
        
        return jsonify(plot_data)
    
    except Exception as e:
        app.logger.error(f"Error generating SHAP summary: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)