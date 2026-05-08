import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

#load dataset - should be in same directory as main.py
data = pd.read_csv("spam_emails_dataset_190.csv")
#error check
if data.empty:
   print("ERROR: dataset not loaded.\n")
   exit(1)
else:
    print("SUCCESS: Dataset loaded.\n")
    print(data.head())

# Remove rows where text is missing
data = data.dropna(subset=["text"])
#dataset should have at least "text" and "label" - convert label to 1/0s
data.loc[data["label"].str.casefold() == "spam", "res"] = 1
data.loc[data["label"].str.casefold() == "ham", "res"] = 0
print(data.head())

#assign X, Y
X = data['text']
Y = data['res']
#split data, 80% train, 20% test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 0.2, random_state = 42)

#tokenization -- TFIDF converts text into numerical features
#parameters: retains all tokens, removes standard stop words from English list, and converts all lowercase
feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': MultinomialNB(),
    'SGD Classifier': SGDClassifier()
}

results = {}
best_model = None
best_f1 = 0

for name, model in models.items():
    # Train the model
    model.fit(X_train_features, Y_train)

    # Predictions
    train_prediction = model.predict(X_train_features)
    test_prediction = model.predict(X_test_features)

    # Calculate metrics
    train_accuracy = accuracy_score(Y_train, train_prediction)
    test_accuracy = accuracy_score(Y_test, test_prediction)
    f1 = f1_score(Y_test, test_prediction, average='weighted')

    # Store results
    results[name] = {
        'Train Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy,
        'F1-Score': f1
    }

    # Track best model
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_test_prediction = test_prediction
        best_name = name

# Display model comparison
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
results_df = pd.DataFrame(results).T
print(results_df)
print(f"\nBest Model: {best_name} (F1 = {best_f1:.3f})")

print("\n" + "="*50)
print("EVALUATION METRICS")
print("="*50)
print(f"Best Model Accuracy on test data: {results[best_name]['Test Accuracy'] * 100:.2f}%")

# Classification Report
print("\nClassification Report (Test Data):")
print(classification_report(Y_test, best_test_prediction, target_names=['ham', 'spam']))

# Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train_features, Y_train, cv=cv, scoring='f1')
print("\nCross-Validation Results:")
print(f"F1 scores across 5 folds: {cv_scores}")
print(f"Mean CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# ROC Curve
y_prob = best_model.predict_proba(X_test_features)[:, 1]
fpr, tpr, _ = roc_curve(Y_test, y_prob)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (AUC = {roc_auc_score(Y_test, y_prob):.3f})')
plt.show()
