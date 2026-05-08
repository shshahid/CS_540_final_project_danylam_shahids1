import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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

#choose logistic regression model
model = LogisticRegression()
#model training
model.fit(X_train_features, Y_train)

#evaluation on train
train_prediction = model.predict(X_train_features)
train_accuracy = accuracy_score(Y_train, train_prediction)
#evaluation on test
test_prediction = model.predict(X_test_features)
test_accuracy = accuracy_score(Y_test, test_prediction)

print("\n" + "="*50)
print("EVALUATION METRICS")
print("="*50)
print(f"Accuracy on training data: {train_accuracy * 100:.2f}%")
print(f"Accuracy on test data: {test_accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report (Test Data):")
print(classification_report(Y_test, test_prediction, target_names=['normal', 'spam']))