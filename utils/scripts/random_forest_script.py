"""
Purpose:
-----------------------------------------------------------------------------------
- Random Forest Classifier Object
- Evaluation (Cross Validation)
-----------------------------------------------------------------------------------
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
import pandas as pd
import numpy as np
import pylab as plt
from clean_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

clf = RandomForestClassifier(n_estimators=100)


df_train = pd.read_csv('../../data/train.csv')
df_test = pd.read_csv('../../data/test.csv')

### Training
train_data = clean_data(df_train)
train_data = train_data.values
# Training data features, skip the first column 'Survived'
train_features = train_data[:, 1:]

# 'Survived' column values
train_target = train_data[:, 0]

# Fit the model to our training data
clf = clf.fit(train_features, train_target)
score = clf.score(train_features, train_target)
"Mean accuracy of Random Forest: {0}".format(score)
#0.98

### Prediction
test_data = clean_data(df_test)
test_data = test_data.values

# Get the test data features, skipping the first column 'PassengerId'
test_x = test_data[:, 0:]

# Predict the Survival values for the test data
test_y = clf.predict(test_x)
# 0.83


### Evaluation
# Cross Validation
# Split 80-20 train vs test data
train_x, test_x, train_y, test_y = train_test_split(train_features, train_target, 
                                                    test_size=0.20, random_state=0)

clf = clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)

print ("Accuracy = %.2f" % (accuracy_score(test_y, predict_y)))
# 0.87

