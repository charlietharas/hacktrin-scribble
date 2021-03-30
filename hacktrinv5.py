'''
Created on Mar 7, 2021

@authors: maggie_kwan
'''


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import cv2
import emnist
from keras.utils import to_categorical
import keras
from pandas import *

print('reading in files...')
train = read_csv('emnist-balanced-train.csv', delimiter=',')
test = read_csv('emnist-balanced-test.csv', delimiter=',')
mapp = read_csv('emnist-byclass-mapping.txt', delimiter=' ', index_col=0, header=None, squeeze=True)
h = 28
w = 28
print('making train/test sets...')
train_x = train.iloc[:,1:]
train_y = train.iloc[:,0]
test_x = test.iloc[:, 1:]
test_y = train.iloc[:,0]
del train
del test


def rte(image):
    image = image.reshape([h, w])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image


train_x = np.asarray(train_x)
train_x = np.apply_along_axis(rte, 1, train_x)
test_x = np.asarray(test_x)
test_x = np.apply_along_axis(rte, 1, test_x)

train_x = train_x.astype('float32')
train_x /= 255
test_x = test_x.astype('float32')
test_x /= 255

num_classes = train_y.unique()
train_y = to_categorical(train_y, num_classes)
test_y = to_categorical(test_y, num_classes)

train_x = train_x.reshape(-1, h, w, 1)
test_x = test_x.reshape(-1, h, w, 1)

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.10, random_state=7)

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(h, w, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=5, batch_size=512, verbose=1, validation_data = (val_x, val_y))

score = model.evaluate(test_x, test_y, verbose=0)
