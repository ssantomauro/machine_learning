#
# This the model I submitted to kaggle. This is my first competition :)
# I hope this script can help who is joining Kaggle for the first time (like me)
#
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from time import time


base_path = os.path.abspath('')

ds_submission_orig = pd.read_csv(base_path+'/datasets/kaggle_digit_recognizer/sample_submission.csv')
ds_test_orig = pd.read_csv(base_path+'/datasets/kaggle_digit_recognizer/test.csv')
ds_train = pd.read_csv(base_path+'/datasets/kaggle_digit_recognizer/train.csv')
X = ds_train.drop(['label'], axis=1)
Y = ds_train['label'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=False)

mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(500,), max_iter=500, random_state=False, verbose=True)
mlp.fit(X_train, Y_train)

# Kaggle submission
X_test_submission = mms.transform(ds_test_orig)
y_pred_submission = mlp.predict(X_test_submission)

submission = pd.DataFrame({'ImageId':ds_submission_orig['ImageId'],'Label':y_pred_submission})
filename = 'kaggle_digit_recognizer_submission.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)
