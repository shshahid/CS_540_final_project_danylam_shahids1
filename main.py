import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve, LearningCurveDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score

# LOAD DATASET - should be in same directory as main.py
data = pd.read_csv('spam_emails_dataset_5.csv')
# Error check
if data.empty:
   print("ERROR: dataset not loaded.\n")
   exit(1)
else:
    print("SUCCESS: Dataset loaded.")
    print(data.head())
    print("\n")

# PREPROCESSING
# Remove rows where text is missing
data = data.dropna(subset=['text'])  
# Dataset should have at least "text" and "label" - convert label to 1/0s
# spam = 1, ham = 0
data.loc[data['label'].str.casefold() == 'spam', 'res'] = 1
data.loc[data['label'].str.casefold() == 'ham', 'res'] = 0
print(data.head())
print("\n")

# Assign X, Y
X = data['text']
Y = data['res']
# SPLIT DATA - 80% train, 20% test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 0.2, random_state = 42
)

# TOKENIZATION -- TFIDF converts text into numerical features
# Parameters: retains all tokens, removes standard stop words from English list, and converts all lowercase
feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# CHOOSE MODELS
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': MultinomialNB(),
    'SGD Classifier': SGDClassifier(loss='log_loss')
}
results = {}
best_model = None
best_f1 = 0
    
# MODEL TRAINING, RESULTS
for name, model in models.items():
    # Train
    model.fit(X_train_features, Y_train)

    # Predictions
    train_prediction = model.predict(X_train_features)
    test_prediction = model.predict(X_test_features)

    # Calculate metrics
    train_accuracy = accuracy_score(Y_train, train_prediction)
    test_accuracy = accuracy_score(Y_test, test_prediction)
    f1 = f1_score(Y_test, test_prediction, average = 'weighted')

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


# EVALUATION
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
results_df = pd.DataFrame(results).T
print(results_df)
print(f"\nBest Model: {best_name} (F1 = {best_f1:.3f})")

print("\n" + "="*50)
print("EVALUATION METRICS")
print("="*50)
print(f"Best Model Train Accuracy on test data: {results[best_name]['Train Accuracy'] * 100:.2f}%")
print(f"Best Model Test Accuracy on test data: {results[best_name]['Test Accuracy'] * 100:.2f}%")

# Classification Report
print("\nClassification Report (Test Data):")
print(classification_report(Y_test, best_test_prediction, target_names=['ham', 'spam']))

# Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train_features, Y_train, cv=cv, scoring='f1')
print("\nCross-Validation Results:")
print(f"F1 scores across 5 folds: {cv_scores}")
print(f"Mean CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Confusion Matrix
conf_matrix = confusion_matrix(Y_test, best_test_prediction)
sns.heatmap(conf_matrix, annot=True, cbar=False, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curves
fig, ax = plt.subplots(1, len(models))
i=0
for name, model in models.items():
    y_prob = model.predict_proba(X_test_features)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, y_prob)
    ax[i].plot(fpr, tpr)
    ax[i].plot([0, 1], [0, 1], '--')
    if(name == best_name):
        ax[i].title.set_text(f'Best model: {name} (AUC = {roc_auc_score(Y_test, y_prob):.3f})')
    else:
        ax[i].title.set_text(f'{name} (AUC = {roc_auc_score(Y_test, y_prob):.3f})')
    ax[i].grid()
    i+=1
fig.suptitle('ROC Curves')
fig.supxlabel('False Positive Rate')
fig.supylabel('True Positive Rate')
plt.show()

# Train learning curve for all models
for name, model in models.items():
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train_features, Y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5
    )
    train_mean = np.mean(train_scores, axis=1)
    plt.plot(train_sizes, train_mean, label=f'{name}')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title('Training Learning Curve on all models')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Test learning curve for all models
for name, model in models.items():
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train_features, Y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5
    )
    test_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, test_mean, label=f"{name}")
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title('Test Learning Curve on all models')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Combined learning curve (model accuracy) for best model
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train_features, Y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5
)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_mean, label='training')
plt.plot(train_sizes, test_mean, label='test')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title(f'Learning Curve on best model: {best_name}')
plt.legend(loc='lower right')

# Alternate plot
#LearningCurveDisplay.from_estimator(
#    best_model, X_train_features, Y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring="accuracy"
#)
plt.grid()
plt.show()