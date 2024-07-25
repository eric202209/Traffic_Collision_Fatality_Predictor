import pickle
from ml_pipeline import load_and_preprocess_data, evaluate_models
from api.app import app
import os
from utils.config import load_config
from utils import setup_logger

def save_pickle(obj, path):
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def main():
    config = load_config()
    logger = setup_logger()
    
    try:
        # Load and preprocess data
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = load_and_preprocess_data(config['data_path'], logger)        
        
        # Evaluate models and get the best one
        best_model = evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, logger)

        # Save model, scaler, and feature names
        save_pickle(best_model, config['model_path'])
        save_pickle(scaler, config['scaler_path'])
        save_pickle(feature_names, config['feature_names_path'])

        # Verify files were created and are not empty
        if all(os.path.getsize(path) > 0 for path in [config['model_path'], config['scaler_path'], config['feature_names_path']]):
            logger.info("Model, scaler, and feature names saved successfully.")
        else:
            logger.error("Error: One or more saved files are empty.")

        logger.info("Starting Flask app...")            
        app.run(debug=config['debug'], port=config['port'])
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

# Go to http://127.0.0.1:5000 or http://localhost:5000 
# (assuming Flask is running on the default port 5000).

