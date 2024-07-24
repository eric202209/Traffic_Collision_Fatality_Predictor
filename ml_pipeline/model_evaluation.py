import matplotlib.pyplot as plt
import seaborn as sns
import os
# from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# def plot_roc_curve(y_true, y_pred_proba, model_name):
#     fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
#     roc_auc = auc(fpr, tpr)
    
#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'Receiver Operating Characteristic (ROC) - {model_name}')
#     plt.legend(loc="lower right")
#     plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()

    # Create a directory for the images if it doesn't exist
    os.makedirs('static/images', exist_ok=True)
    
    # Save the figure instead of showing it
    plt.savefig(f'static/images/confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    plt.close()  # Close the figure to free up memory

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    # Cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
   
    print(f"\n{model_name} Results:")
    print(f"Cross-Validation Mean Accuracy: {cv_scores.mean()}")
   
    # Training and evaluation on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
   
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name)

    return model

def evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    best_model = None
    best_accuracy = 0

    # Train and evaluate models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Support Vector Machine": SVC(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        # "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        "Neural Network": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, learning_rate_init=0.001, early_stopping=True, tol=1e-3, random_state=42)
    }

    for name, model in models.items():
        # if hasattr(model, "predict_proba"):
        #     y_pred_proba = model.predict_proba(X_test)[:, 1]
        #     plot_roc_curve(y_test, y_pred_proba, name)
        # else:
        #     print(f"ROC curve not available for {name}")

        if name in ["Logistic Regression", "Support Vector Machine", "Neural Network"]:
            trained_model = train_and_evaluate(model, X_train_scaled, X_test_scaled, y_train, y_test, name)
        else:
            trained_model = train_and_evaluate(model, X_train, X_test, y_train, y_test, name)
        
        # Check if this model is the best so far
        y_pred = trained_model.predict(X_test_scaled if name in ["Logistic Regression", "Support Vector Machine", "Neural Network"] else X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = trained_model

    return best_model

