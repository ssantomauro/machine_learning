#
# This the model I submitted to kaggle. This is my first competition :)
# I hope this script can help who is joining Kaggle for the first time (like me)
#
from pathlib import Path
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from xgboost import XGBClassifier

base_path = os.path.abspath('')

ds_train_orig = pd.read_csv(base_path+'/datasets/kaggle_titanic/train.csv')
ds_survived_orig = pd.read_csv(base_path+'/datasets/kaggle_titanic/gender_submission.csv')
ds_test_orig = pd.read_csv(base_path+'/datasets/kaggle_titanic/test.csv')

ds_test_merged = pd.merge(ds_survived_orig, ds_test_orig, on='PassengerId')

ds_full = ds_train_orig.append(ds_test_merged)

le = LabelEncoder()
ds_train_orig['Sex'] = le.fit_transform(ds_train_orig['Sex'])
ds_train_orig.Age = ds_train_orig.Age.fillna(ds_full.Age.mean())
# I'm filling Fare NaN using the median from the corresponding Pclass
# calculated on the full dataset (train + test)
ds_train_orig.Fare = ds_train_orig.Fare.fillna(ds_full.groupby('Pclass').median().Fare[int(ds_full.loc[(ds_full.Fare.isnull())].Pclass)])

ds_test_merged['Sex'] = le.fit_transform(ds_test_merged['Sex'])
ds_test_merged.Age = ds_test_merged.Age.fillna(ds_full.Age.mean())
ds_test_merged.Fare = ds_test_merged.Fare.fillna(ds_full.groupby('Pclass').mean().Fare[int(ds_full.loc[(ds_full.Fare.isnull())].Pclass)])

X_train = ds_train_orig.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked','SibSp', 'Survived'], axis=1)
Y_train = ds_train_orig['Survived'].values

X_test = ds_test_merged.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked','SibSp', 'Survived'], axis=1)
Y_test = ds_test_merged['Survived'].values

from sklearn.model_selection import GridSearchCV

# I've already set the best params found
parameters = {
    'n_estimators'      : [200],
    'max_depth'         : [3],
    'random_state'      : [0],
    'learning_rate': [0.1],
}

clf = GridSearchCV(XGBClassifier(), parameters, cv=10, n_jobs=-1, scoring='accuracy')
clf.fit(X_train, Y_train)

y_pred_train = clf.predict(X_train)
y_pred = clf.predict(X_test)

accuracy_train = accuracy_score(Y_train, y_pred_train)
accuracy_test = accuracy_score(Y_test, y_pred)
print("ACCURACY: TRAIN=%.4f TEST=%.4f" % (accuracy_train,accuracy_test))

submission = pd.DataFrame({'PassengerId':ds_test_merged['PassengerId'],'Survived':y_pred})
filename = 'kaggle_titanic_submission.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)
