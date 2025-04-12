import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import adam

# Input layer with shape (28, 1)
inputs = layers.Input(shape=(28, 1))
x = inputs

x = layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
x = layers.Dense(units=10, activation='softmax')

# Build model
model = tf.keras.Model(inputs=inputs, outputs=x)
# Compile model with adam optimizer and categorical_crossentropy loss
model.compile(loss='categorical_crossentropy', optimizer=adam())
