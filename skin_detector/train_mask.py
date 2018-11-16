from __future__ import print_function
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split


df = pd.read_csv('~/Desktop/datasets/Skin_NonSkin.txt', delimiter='\t', names=['R','G','B','is_skin'])
X = df[['R','G','B']]
Y = df['is_skin']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
Y_train = keras.utils.to_categorical(Y_train)
Y_test = keras.utils.to_categorical(Y_test)
Y_train = np.delete(Y_train, [0], axis=1)
Y_test = np.delete(Y_test, [0], axis=1)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(3,)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=128, epochs=5, verbose=1, validation_split=0.1)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Loss:',score[0])
print('Accuracy:',score[1])

model.save('mask.h5')