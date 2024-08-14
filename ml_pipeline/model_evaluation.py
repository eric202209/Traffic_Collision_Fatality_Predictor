import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from utils.logger import setup_logger
import shap
import pickle
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from scipy.stats import randint, uniform
from sklearn.preprocessing import StandardScaler


logger = setup_logger()

def plot_confusion_matrix(y_true, y_pred, model_name, logger):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Create a directory for the images if it doesn't exist
    os.makedirs('static/images', exist_ok=True)
    
    # Save the figure instead of showing it
    plt.savefig(f'static/images/confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    plt.close()  # Close the figure to free up memory
    logger.info(f"Confusion matrix for {model_name} saved")

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, logger):
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
   
    logger.info(f"\n{model_name} Results:")
    logger.info(f"Cross-Validation Mean Accuracy: {cv_scores.mean()}")
   
    # Training and evaluation on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted')
    }
    
    logger.info(f"Test Set Metrics: {metrics}")
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))
   
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name, logger)
    
    logger.info(f"{model_name} evaluation completed")
    return model, metrics

def perform_randomized_search(model, param_distributions, X, y, cv, n_iter=50):
    random_search = RandomizedSearchCV(
        model, 
        param_distributions, 
        n_iter=n_iter, 
        cv=cv, 
        scoring='accuracy', 
        n_jobs=-1, 
        random_state=42,
    )
    random_search.fit(X, y)
    return random_search.best_estimator_

def plot_shap_summary(shap_values, X_test_scaled, feature_names):
    plt.figure(figsize=(10, 8))
    if isinstance(shap_values, list):
        # For classification problems
        for i, sv in enumerate(shap_values):
            shap.summary_plot(sv, X_test_scaled, feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary for Class {i}")
            plt.tight_layout()
            plt.savefig(f'static/images/shap_summary_class_{i}.png')
            plt.close()
    else:
        shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('static/images/shap_summary.png')
        plt.close()
    logger.info("SHAP summary plot saved.")

def evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, logger):
    logger.info("Starting model evaluation...")
    
    # Define parameter distributions for RandomizedSearchCV
    param_distributions = {
        "Logistic Regression": {
            'C': uniform(0.1, 10),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        "Decision Tree": {
            'max_depth': randint(1, 32),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20)
        },
        "Support Vector Machine": {
            'C': uniform(0.01, 100),
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'] + list(uniform(0.0001, 0.1).rvs(10)),
            'tol': uniform(1e-6, 1e-2),
            'max_iter': [2000, 5000, 10000]
        },
        "Random Forest": {
            'n_estimators': randint(10, 200),
            'max_depth': randint(1, 32),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20)
        },
        "Neural Network": {
            'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
            'activation': ['tanh', 'relu'],
            'alpha': uniform(0.0001, 0.1),
            'learning_rate': ['constant', 'adaptive']
        },
        "XGBoost": {
            'n_estimators': randint(10, 200),
            'max_depth': randint(1, 32),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5)
        }
    }

    base_models = {
        "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Support Vector Machine": SVC(random_state=42, class_weight='balanced', probability=True),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "Neural Network": MLPClassifier(max_iter=1000, early_stopping=True, random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
    }

    logger.info("Applying SMOTETomek for data balancing...")
    
    try:
        smote_tomek = SMOTETomek(random_state=42)
        X_resampled, y_resampled = smote_tomek.fit_resample(X_train_scaled, y_train)
        logger.info("SMOTETomek applied successfully.")

        # Scale the data after resampling
        scaler = StandardScaler()
        X_resampled_scaled = scaler.fit_transform(X_resampled)
    
    except Exception as e:
        logger.error(f"Error in applying SMOTETomek: {str(e)}")
        X_resampled, y_resampled = X_train_scaled, y_train

    logger.info("Starting model training and evaluation...")
    best_models = {}
    best_metrics = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in base_models.items():
        logger.info(f"Tuning hyperparameters for {name}...")
        try:
            best_model = perform_randomized_search(model, param_distributions[name], X_resampled_scaled, y_resampled, cv)
            trained_model, metrics = train_and_evaluate(best_model, X_resampled_scaled, X_test_scaled, y_resampled, y_test, name, logger)
            
            best_models[name] = trained_model
            best_metrics[name] = metrics
        except Exception as e:
            logger.error(f"Error in evaluating {name}: {str(e)}")
    
    # Create and evaluate VotingClassifier
    logger.info("Creating and evaluating VotingClassifier...")
    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in best_models.items()],
        voting='soft'
    )
    
    voting_model, voting_metrics = train_and_evaluate(voting_clf, X_resampled_scaled, X_test_scaled, y_resampled, y_test, "Voting Classifier", logger)
    best_models["Voting Classifier"] = voting_model
    best_metrics["Voting Classifier"] = voting_metrics

    # Find the best overall model
    best_model_name = max(best_metrics, key=lambda x: best_metrics[x]['Accuracy'])
    best_model = best_models[best_model_name]
    best_metric = best_metrics[best_model_name]

    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Best model metrics: {best_metric}")
    
    logger.info("Saving best model and metrics...")
    try:
        with open('models/best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        with open('models/best_model_metrics.pkl', 'wb') as f:
            pickle.dump(best_metric, f)
        logger.info("Best model and metrics saved successfully.")
    except Exception as e:
        logger.error(f"Error in saving best model: {str(e)}")

    logger.info("Calculating or loading SHAP values...")
    try:
        # Reduce the number of background samples
        background_data = shap.kmeans(X_train_scaled, 100)  # Use 100 background samples

        # Use a subset of your test data
        X_test_subset = X_test_scaled[:100]  # Use first 100 samples, or use random sampling

        # Use a subset of your test data, randomly sampled for better representation
        np.random.seed(42)  # For reproducibility
        X_test_subset_indices = np.random.choice(X_test_scaled.shape[0], 100, replace=False)
        X_test_subset = X_test_scaled[X_test_subset_indices]

        if hasattr(best_model, 'predict_proba'):
            if hasattr(best_model, 'feature_importances_'):
                explainer = shap.TreeExplainer(best_model, data=background_data)
            else:
                explainer = shap.KernelExplainer(best_model.predict_proba, background_data, 
                                                 link="logit")  # Specify link function for better performance     
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test_subset, nsamples=100)  # Limit the number of samples for faster computation
            
            # Save SHAP values
            with open('models/best_model_shap_values.pkl', 'wb') as f:
                pickle.dump(shap_values, f)
            logger.info("SHAP values calculated and saved successfully.")

            # Plot and save SHAP summary
            plot_shap_summary(shap_values, X_test_subset)

    except Exception as e:
        logger.error(f"Error in calculating SHAP values: {str(e)}")
    
    logger.info("Model evaluation process completed.")
    return best_model, scaler

# Save the feature_names
def save_feature_names(feature_names):
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

# Save the scaler
def save_scaler(scaler):
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
