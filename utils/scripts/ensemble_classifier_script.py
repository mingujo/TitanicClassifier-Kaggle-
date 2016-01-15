"""
Purpose:
-----------------------------------------------------------------------------------
- Ensemble Classifier (Gradient Boosting & Logistic Regression)
-----------------------------------------------------------------------------------
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))
import pandas as pd
import numpy as np
import pylab as plt
from clean_data import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold

df_train = pd.read_csv('../../data/train.csv')
df_test = pd.read_csv('../../data/test.csv')

### Training
train_data_frame = clean_data(df_train,drop_passenger_id=True)
train_data = train_data_frame.values
# Training data features, skip the first column 'Survived'
train_features = train_data[:, 1:]


# # which features are the best?
predictors = ["Pclass", "Fare", "Sex", "Ticket","Embarked_Val_C","Embarked_Val_Q", "Embarked_Val_S", 
			"FamilySize", "AgeFill","AgeCat","Fare_per_person", "Title", "HighLow","FamilyId", \
			"Age_class", "Fare_class","Family", \
			#"Sex_class","AgeFill_squared","Age_class_squared",\
			]

### Ensemble Model
algorithms = [
    [GradientBoostingClassifier(learning_rate=0.005, n_estimators=250,
                                max_depth=10, subsample=0.5,
                                max_features=0.5,random_state=1), predictors],
    [LogisticRegression(random_state=1), predictors]
]
#does ordering matter for logistic regression?


# initialize k-fold cross validation
kf = KFold(train_data_frame.shape[0], n_folds=3, random_state=1)

# Train ensemble model
predictions = []
for train, test in kf:
    train_target = train_data_frame["Survived"].loc[train]
    all_test_predictions = []
    # make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # fit the algorithm on the training data.
        alg.fit(train_data_frame[predictors].iloc[train,:], train_target)
        # select and predict on the test fold.  
        test_predictions = alg.predict_proba(train_data_frame[predictors].iloc[test,:].astype(float))[:,1]
        # .astype(float) is necessary for sklearn
        all_test_predictions.append(test_predictions)
    # just average the predictions to get the final classification.
    # gradient boosting classifier generates better predictions, so we weight it higher.
    test_predictions = (all_test_predictions[0]*3 + all_test_predictions[1]) / 4
    # any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
accuracy = sum(predictions[predictions == train_data_frame["Survived"]]) / len(predictions)
print(accuracy)
# accuracy = 0.835


# Run model on Test set

test_data_frame = clean_data(df_test,drop_passenger_id=False)
test_data = test_data_frame.values
# Training data features, skip the first column 'Survived'
test_features = test_data[:, 1:]

all_predictions = []
for alg, predictors in algorithms:
    # fit the algorithm using the training data.
    alg.fit(train_data_frame[predictors], train_data_frame["Survived"])
    # predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(test_data_frame[predictors].astype(float))[:,1]
    all_predictions.append(predictions)

predictions = (all_predictions[0] * 5 + all_predictions[1]) / 6
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
# accuracy = 0.84

predictions = predictions.astype(int)
df_test['Survived'] = predictions
df_test[['PassengerId', 'Survived']] \
    .to_csv('../../data/results-ensemble.csv', index=False)

