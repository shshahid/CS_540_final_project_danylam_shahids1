import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#load dataset - should be in same directory as main.py
data = pd.read_csv("spam_ham_dataset.csv")
if data.empty:
   print("ERROR: dataset not loaded.\n")
   exit(1)
else:
    print("SUCCESS: Dataset loaded.\n")
    print(data.head())

X = data['text']
Y = data['label_num']
#80% train, 20% test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#TFIDF convert text into numerical feature - 
#parameters: retains all tokens, removes standard stop words, and converts all lowercase
feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

model = LogisticRegression()
model.fit(X_train_features, Y_train)

train_prediction = model.predict(X_train_features)
train_accuracy = accuracy_score(Y_train, train_prediction)

test_prediction = model.predict(X_test_features)
test_accuracy = accuracy_score(Y_test, test_prediction)

print("Accuracy on training data: {}%".format(train_accuracy * 100))
print("Accuracy on test data: {}%".format(test_accuracy * 100))