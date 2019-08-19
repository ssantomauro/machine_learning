#
# This is a test inspired by https://www.kaggle.com/atuljhasg
# As you can see comparing the 'kaggle_titanic_model.py' and this file
# even if this one has a more precise dataset, the 'kaggle_titanic_model.py'
# has a better accuracy
#
import math
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

base_path = os.path.abspath('')

def encode_features(df_train, df_test):
    features = ['Title','Embarked','Ticket']

    df_combined = pd.concat([df_train[features], df_test[features]])

    for feature in features:
        le = LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

def normalize_title(df):

    df["Title"] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False) #Creating new column name Title

    df["Title"] = df["Title"].replace('Master', 'Master')
    df["Title"] = df["Title"].replace('Mlle', 'Miss')
    df["Title"] = df["Title"].replace(['Mme', 'Dona', 'Ms'], 'Mrs')
    df["Title"] = df["Title"].replace(['Don','Jonkheer'],'Mr')
    df["Title"] = df["Title"].replace(['Capt','Rev','Major', 'Col','Dr'], 'Millitary')
    df["Title"] = df["Title"].replace(['Lady', 'Countess','Sir'], 'Honor')

    return df

def normalize_age(df, df_full):
    titles = ['Master', 'Miss', 'Mr', 'Mrs', 'Millitary','Honor']
    for title in titles:
        age_to_impute = df_full.groupby('Title')['Age'].median()[title]
        df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = age_to_impute

    return df


ds_train_orig = pd.read_csv(base_path+'/datasets/kaggle_titanic/train.csv')
ds_survived_orig = pd.read_csv(base_path+'/datasets/kaggle_titanic/gender_submission.csv')
ds_test_orig = pd.read_csv(base_path+'/datasets/kaggle_titanic/test.csv')

ds_test_merged = pd.merge(ds_survived_orig, ds_test_orig, on='PassengerId')
ds_full = ds_train_orig.append(ds_test_merged)
ds_test_merged[['PassengerId','Name','Survived']]

ds_full = normalize_title(ds_full)
ds_train_orig = normalize_title(ds_train_orig)
ds_test_merged = normalize_title(ds_test_merged)

#Fill the na values in Fare
ds_train_orig['Embarked'] = ds_train_orig['Embarked'].fillna('S') #NAN Values set to S class
ds_test_merged['Embarked'] = ds_test_merged['Embarked'].fillna('S') #NAN Values set to S class

# convert Embarked categories to Columns
ds_train_orig = pd.get_dummies(ds_train_orig, columns=['Embarked'])
ds_test_merged = pd.get_dummies(ds_test_merged, columns=['Embarked'])


le = LabelEncoder()
ds_train_orig['Sex'] = le.fit_transform(ds_train_orig['Sex'])
ds_train_orig = normalize_age(ds_train_orig, ds_full)
ds_train_orig.Fare = ds_train_orig.Fare.fillna(ds_full.groupby('Pclass').mean().Fare[int(ds_full.loc[(ds_full.Fare.isnull())].Pclass)])

ds_test_merged['Sex'] = le.fit_transform(ds_test_merged['Sex'])
ds_test_merged = normalize_age(ds_test_merged, ds_full)
ds_test_merged.Fare = ds_test_merged.Fare.fillna(ds_full.groupby('Pclass').mean().Fare[int(ds_full.loc[(ds_full.Fare.isnull())].Pclass)])

# now I can remove Title columns because already used to normalize Age
ds_train_orig = pd.get_dummies(ds_train_orig, columns=['Title'])
ds_test_merged = pd.get_dummies(ds_test_merged, columns=['Title'])
ds_test_merged.insert(14, 'Title_Honor', 0) #I add this columns because does not exist in the original test dataset

X_train = ds_train_orig.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Survived'], axis=1)
Y_train = ds_train_orig['Survived'].values

X_test = ds_test_merged.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Survived'], axis=1)
Y_test = ds_test_merged['Survived'].values

# RandomForestClassifier
# If you want to play a little with this algorithm remember to change the GridSearchCV param
# parameters = {
#     'n_estimators'      : [10,15,20,25,30,35],
#     'max_depth'         : [1,3,5,7,9],
#     'random_state'      : [False],
# }

# DecisionTreeClassifier
# If you want to play a little with this algorithm remember to change the GridSearchCV param
# parameters = {
#     'criterion'      : ['entropy','gini'],
#     'max_depth'         : [1,3,5,7,9],
#     'random_state'      : [False],
# }

# XGBClassifier: I've already set the best params found
parameters = {
    'n_estimators'      : [115],
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

# submission = pd.DataFrame({'PassengerId':ds_test_merged['PassengerId'],'Survived':y_pred})
# filename = 'titanic_predictions_3.csv'
# submission.to_csv(filename,index=False)
# print('Saved file: ' + filename)
