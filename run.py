import pickle
from ml_pipeline import load_and_preprocess_data, evaluate_models
from api.app import app
import os

def main():
    try:
        # Load and preprocess data
        file_path = 'D:/centennial/centennial 2024 summer/comp247/Assignment/Group Project/Traffic_Collision_Fatality_Predictor/Killed_and_Seriously_Injured.csv'
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = load_and_preprocess_data(file_path)        # Evaluate models and get the best one
        
        # Evaluate models and get the best one
        best_model = evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)

        # Save model and scaler
        with open('models/best_model.pkl', 'wb') as file:
            pickle.dump(best_model, file)
        with open('models/scaler.pkl', 'wb') as file:
            pickle.dump(scaler, file)

        # Save feature names
        with open('models/feature_names.pkl', 'wb') as file:
            pickle.dump(feature_names, file)

        # Verify files were created and are not empty
        if all(os.path.getsize(f'models/{file}.pkl') > 0 for file in ['best_model', 'scaler', 'feature_names']):
            print("Model, scaler, and feature names saved successfully.")
        else:
            print("Error: One or more saved files are empty.")
            
        print("Model trained and saved. Starting Flask app...")
        app.run(debug=True)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

# Go to http://127.0.0.1:5000 or http://localhost:5000 
# (assuming Flask is running on the default port 5000).

