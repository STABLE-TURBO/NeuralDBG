import tensorflow as tf
from tensorflow import keras

model = keras.Sequential(name='MyModel', layers=[
    keras.layers.Flatten(input_shape=(None, 28, 28)),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(units=10, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='Adam')
