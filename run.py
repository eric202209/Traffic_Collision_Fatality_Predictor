import pickle
from ml_pipeline import load_and_preprocess_data, evaluate_models
from api.app import app
import os
from utils.config import load_config
from utils import setup_logger
import pandas as pd
import traceback

def save_pickle(obj, path):
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def main():
    os.makedirs('models', exist_ok=True)
    config = load_config()
    logger = setup_logger()
    
    try:
        # Load and preprocess data
        logger.info(f"Attempting to load data from {config['data_path']}")
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = load_and_preprocess_data(config['data_path'], logger)

        # Log the shape of the data and the first few feature names
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"First few feature names: {feature_names[:5]}")

        # Evaluate models and get the best one
        best_model = evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, logger)
        
        # Save model, scaler, and feature names
        save_pickle(best_model, config['model_path'])
        save_pickle(scaler, config['scaler_path'])
        save_pickle(feature_names, config['feature_names_path'])

        # Verify files were created and are not empty
        files_to_check = [
            config['model_path'],
            config['scaler_path'],
            config['feature_names_path'],
            'models/best_model_shap_values.pkl'
        ]
        if all(os.path.exists(path) and os.path.getsize(path) > 0 for path in files_to_check):
            logger.info("Model, scaler, feature names, and SHAP values saved successfully.")
        else:
            logger.error("Error: One or more saved files are missing or empty.")

        # Verify files were created and are not empty
        if all(os.path.getsize(path) > 0 for path in [config['model_path'], config['scaler_path'], config['feature_names_path']]):
            logger.info("Model, scaler, and feature names saved successfully.")
        else:
            logger.error("Error: One or more saved files are empty.")

        logger.info("Starting Flask app...")            
        app.run(debug=config['debug'], port=config['port'])
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        print(f"Error: {str(e)}")
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error: {str(e)}")
        print(f"Error: The data file is empty or contains no data.")
    except pd.errors.ParserError as e:
        logger.error(f"Parser error: {str(e)}")
        print(f"Error: There was a problem parsing the CSV file. Please check its format.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"An unexpected error occurred: {str(e)}")
        logger.exception("Stack trace:")
        print("Please check the log file for more details.")

if __name__ == "__main__":
    main()

# Go to http://127.0.0.1:5000 or http://localhost:5000 
# (assuming Flask is running on the default port 5000).

