import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from utils.logger import setup_logger
import shap
import pickle

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
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
   
    logger.info(f"\n{model_name} Results:")
    logger.info(f"Cross-Validation Mean Accuracy: {cv_scores.mean()}")
   
    # Training and evaluation on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Test Set Accuracy: {accuracy:.4f}")
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))

    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    
    logger.info(f"Test Set Metrics: {metrics}")
    logger.info("\nClassification Report:")
    logger.info("\n"+ classification_report(y_test, y_pred))
   
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name, logger)
    
    logger.info(f"{model_name} evaluation completed")
    return model, metrics

def evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, logger):
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Support Vector Machine": SVC(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        # "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        "Neural Network": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, learning_rate_init=0.001, early_stopping=True, tol=1e-3, random_state=42)
    }

    best_model = None
    best_metrics = None
    best_accuracy = 0

    for name, model in models.items():
        if name in ["Logistic Regression", "Support Vector Machine", "Neural Network"]:
            trained_model, metrics = train_and_evaluate(model, X_train_scaled, X_test_scaled, y_train, y_test, name, logger)
        else:
            trained_model, metrics = train_and_evaluate(model, X_train, X_test, y_train, y_test, name, logger)

        if metrics['Accuracy'] > best_accuracy:
            best_accuracy = metrics['Accuracy']
            best_model = trained_model
            best_metrics = metrics

    # Save the best model and its metrics
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('models/best_model_metrics.pkl', 'wb') as f:
        pickle.dump(best_metrics, f)    
    
    # Calculate and save SHAP values for the best model
    if hasattr(best_model, 'predict_proba'):
        explainer = shap.TreeExplainer(best_model) if hasattr(best_model, 'feature_importances_') else shap.KernelExplainer(best_model.predict_proba, X_test_scaled)
        shap_values = explainer.shap_values(X_test_scaled)
        with open('models/best_model_shap_values.pkl', 'wb') as f:
            pickle.dump(shap_values, f)
            
    logger.info("Model evaluation completed")
    return best_model

# Save the feature_names
def save_feature_names(feature_names):
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

# Save the scaler
def save_scaler(scaler):
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
