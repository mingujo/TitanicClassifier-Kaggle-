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
from sklearn import metrics, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold

clf = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=10, min_samples_leaf=8)

df_train = pd.read_csv('../../data/train.csv')
df_test = pd.read_csv('../../data/test.csv')

### Training
train_data_frame = clean_data(df_train,drop_passenger_id=True)
train_data = train_data_frame.values
# Training data features, skip the first column 'Survived'
train_features = train_data[:, 1:]


### feature selection
# which features are the best?
predictors = ["Pclass", "Fare", "Sex_Val", "Embarked_Val", "FamilySize", "Fare_per_person", "AgeFill", "Title", "FamilyId"]
selector = SelectKBest(f_classif, k=5)
selector.fit(train_data_frame[predictors], train_data_frame["Survived"])
scores = -np.log10(selector.pvalues_)
# best features are (most to least): Sex_Val, Title, Pclass, Fare, Fare_per_person, FamilySize
predictors_log = ["Pclass", "Fare", "Sex_Val", "Fare_per_person", "FamilySize"]
scores = cross_validation.cross_val_score(clf, df_train[predictors], df_train["Survived"], cv=3)

### Ensemble Model
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), predictors_log]
]
#does ordering matter for logistic regression?

# initialize k-fold cross validation
kf = KFold(df_train.shape[0], n_folds=3, random_state=1)

# Train ensemble model
predictions = []
for train, test in kf:
    train_target = df_train["Survived"].loc[train]
    all_test_predictions = []
    # make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # fit the algorithm on the training data.
        alg.fit(df_train[predictors].iloc[train,:], train_target)
        # select and predict on the test fold.  
        test_predictions = alg.predict_proba(df_train[predictors].iloc[test,:].astype(float))[:,1]
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
accuracy = sum(predictions[predictions == df_train["Survived"]]) / len(predictions)
print(accuracy)
# accuracy = 0.835


# Run model on Test set



predictions = predictions.astype(int)
df_test['Survived'] = predictions
df_test[['PassengerId', 'Survived']] \
    .to_csv('../../data/results-ensemble.csv', index=False)

