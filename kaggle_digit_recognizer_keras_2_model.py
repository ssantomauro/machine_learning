from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import math


def custom_reshape(arr, rows, cols):
    t = np.zeros(shape=(arr.shape[0], rows, cols))
    for i in range(0,arr.shape[0]):
        temp = arr[i]
        t[i] = temp.reshape(rows, cols)

    return t


batch_size = 100
num_classes = 10
epochs = 25
img_rows, img_cols = 28, 28


base_path = os.path.abspath('')

ds_submission_orig = pd.read_csv(base_path+'/datasets/kaggle_digit_recognizer/sample_submission.csv')
ds_test_orig = pd.read_csv(base_path+'/datasets/kaggle_digit_recognizer/test.csv')
ds_train = pd.read_csv(base_path+'/datasets/kaggle_digit_recognizer/train.csv')
X = ds_train.drop(['label'], axis=1)
Y = ds_train['label'].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=False)

mms = MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_test = mms.transform(x_test)

x_train = custom_reshape(x_train, img_rows, img_cols)
x_test = custom_reshape(x_test, img_rows, img_cols)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(GlobalAveragePooling2D())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Kaggle submission
X_test_submission = mms.transform(ds_test_orig)
X_test_submission = custom_reshape(X_test_submission, img_rows, img_cols)
X_test_submission = X_test_submission.reshape(X_test_submission.shape[0], img_rows, img_cols, 1)

y_pred_submission = model.predict(X_test_submission)
y_pred_submission = list(map(lambda row: np.where(row == np.amax(row))[0], y_pred_submission))
y_pred_submission = list(map(lambda i: i[0] if len(i) > 0 else -1, y_pred_submission))

submission = pd.DataFrame({'ImageId':ds_submission_orig['ImageId'],'Label':y_pred_submission})
filename = 'kaggle_digit_recognizer_keras_2_submission.csv'
submission.to_csv(filename, index=False)
print('Saved file: ' + filename)
