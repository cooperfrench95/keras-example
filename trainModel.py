import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout, Lambda


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


IMG_SIZE = 64  # was 64
AMOUNT_CLASSES = 460

# Import data
features = None
labels = None

with open('features.pickle', 'rb') as f:
    features = pickle.load(f)

with open('labels.pickle', 'rb') as l:
    labels = pickle.load(l)

# Split into training and validation sets
features_train = features[:-10000]
features_test = features[-10000:]
labels_train = labels[:-10000]
labels_test = labels[-10000:]

# added by BFW
model = Sequential()

# # Build model
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

### 1 ###
# Convolution
# input shape is one dimensional
x = Conv2D(32, (3, 3), input_shape=features.shape[1:],
           activation="relu", use_bias=True, padding="same")(inputs)
# Pooling
x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
# Dropout to reduce overfit
x = Dropout(0)(x)

### 2 ###
# Convolution
x = Conv2D(64, (3, 3), activation="linear", use_bias=True, padding="same")(x)
# Pooling
x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
# Dropout to reduce overfit
x = Dropout(0.0)(x)

### 3 ###
# Convolution
x = Conv2D(128, (3, 3), activation="linear", use_bias=True, padding="same")(x)
# Pooling
x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
# Dropout to reduce overfit
x = Dropout(0.3)(x)

### 4 ###
# Convolution
x = Conv2D(256, (3, 3), activation="linear", use_bias=True, padding="same")(x)
# Pooling
x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
# Dropout to reduce overfit
x = Dropout(0.45)(x)

### 5 ###
# Convolutions
x = Conv2D(512, (3, 3), activation="linear", use_bias=True, padding="same")(x)
# Pooling
x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
# Flatten
x = Flatten()(x)
# Dropout to reduce overfit
x = Dropout(0.65)(x)


### 6 ###
# Final
x = Dense(512, activation="linear")(x)

### 7 ###
# Output layer
preds = Dense(AMOUNT_CLASSES, activation="softmax")(x)
model = Model(inputs=inputs, outputs=preds)


# overview of the model components
model.summary()

# Train model
model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    # loss=tf.keras.losses.categorical_crossentropy,
    # learning_rate=0.0009 added learning rate to lower loss
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[
        # tf.keras.metrics.SparseCategoricalAccuracy(),
        'accuracy'
    ]
)


history = model.fit(features_train, labels_train,
                    batch_size=100, validation_split=0.3, epochs=40)

results = model.evaluate(features_test, labels_test, batch_size=100)

print('test loss, test acc:', results)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
