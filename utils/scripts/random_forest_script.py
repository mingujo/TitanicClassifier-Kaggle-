"""
Purpose:
-----------------------------------------------------------------------------------
- Random Forest Classifier Object
- Evaluation (Cross Validation)
-----------------------------------------------------------------------------------
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))
import pandas as pd
import numpy as np
import pylab as plt
from clean_data import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

clf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, min_samples_split=1,
  min_samples_leaf=1, max_features='auto',    bootstrap=False, oob_score=False, n_jobs=1, random_state=1,
  verbose=0)

df_train = pd.read_csv('../../data/train.csv')
df_test = pd.read_csv('../../data/test.csv')

### Training
train_data_frame = clean_data(df_train,drop_passenger_id=True)
train_data = train_data_frame.values
# Training data features, skip the first column 'Survived'
train_features = train_data[:, 1:]

# ### feature selection
# # which features are the best?
# predictors = ["Pclass", "Fare", "Sex_Val", "Embarked_Val", "FamilySize", \
# 				"Fare_per_person", "AgeFill", "Title", "FamilyId"]
# selector = SelectKBest(f_classif, k=5)
# selector.fit(train_data_frame[predictors], train_data_frame["Survived"])
# scores = -np.log10(selector.pvalues_)
# # best features are (most to least): Sex_Val, Title, Pclass, Fare, Fare_per_person, FamilySize


# 'Survived' column values
train_target = list(train_data[:, 0])

# Fit the model to our training data
clf = clf.fit(train_features, train_target)
score = clf.score(train_features, train_target)
"Mean accuracy of Random Forest: {0}".format(score)
#0.98

### Prediction
test_data_frame = clean_data(df_test,drop_passenger_id=False)
test_data = test_data_frame.values

# Get the test data features, skipping the first column 'PassengerId'
test_x = test_data[:, 1:]

# Predict the Survival values for the test data
test_y = clf.predict(test_x).astype(int)
# 0.83
df_test['Survived'] = test_y
df_test[['PassengerId', 'Survived']] \
    .to_csv('../../data/results-rf.csv', index=False)

### Evaluation
# Cross Validation
# Split 80-20 train vs test data
train_x, test_x, train_y, test_y = train_test_split(train_features, train_target, 
                                                    test_size=0.20, random_state=0)

clf = clf.fit(train_x, train_y)
predict_y = clf.predict(test_x).astype(int)

print ("Accuracy = %.5f" % (accuracy_score(test_y, predict_y)))


### Fit to Test Set

# 0.87

