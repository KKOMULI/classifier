import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 

paths = glob.glob("E:\\MachineLearning\\discriminator\\gender\\preprocessed\\*\\*.jpg")
paths = np.random.permutation(paths)
indipendent_v = np.array([plt.imread(paths[i]) for i in range(len(paths))])
subordination_v = np.array([paths[i].split('\\')[-2] for i in range(len(paths))])
print(indipendent_v.shape, subordination_v.shape)

subordination_v = pd.get_dummies(subordination_v)
print(indipendent_v.shape, subordination_v.shape)

X = tf.keras.layers.Input(shape=[100, 100, 3])

H = tf.keras.layers.Conv2D(6, kernel_size=5, padding='same', activation='swish')(X)
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Flatten()(H)
H = tf.keras.layers.Dense(120, activation='swish')(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)

Y = tf.keras.layers.Dense(2, activation='softmax')(H)

model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(indipendent_v, subordination_v, epochs=10)

pred = model.predict(indipendent_v[0:5])
print(pd.DataFrame(pred).round(2))

print(subordination_v[0:5])
