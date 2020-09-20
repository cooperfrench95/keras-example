import pickle
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout, Lambda

IMG_SIZE=64
AMOUNT_CLASSES=460

# Import data
features = None
labels = None

with open('training_no_mask_features.pickle', 'rb') as f:
  features = pickle.load(f)

with open('training_no_mask_labels.pickle', 'rb') as f:
  labels = pickle.load(f)

# Split into training and validation sets
features_train = features[:-10000]
features_test = features[-10000:]
labels_train = labels[:-10000]
labels_test = labels[-10000:]


# # Build model
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
# Convolution
x = Conv2D(32, (3, 3), input_shape=features.shape[1:], activation="relu", use_bias=False, padding="same")(inputs)
# Pooling
x = MaxPooling2D(pool_size=(2,2), padding="same")(x)
# Dropout to reduce overfit
# x = Dropout(0.3)(x)
# Convolution
x = Conv2D(64, (3, 3), activation="linear", use_bias=False, padding="same")(x)
# Pooling
x = MaxPooling2D(pool_size=(2,2), padding="same")(x)
# Dropout to reduce overfit
# x = Dropout(0.3)(x)
# Convolution
x = Conv2D(128, (3, 3), activation="linear", use_bias=False, padding="same")(x)
# Pooling
x = MaxPooling2D(pool_size=(2,2), padding="same")(x)
# Dropout to reduce overfit
x = Dropout(0.3)(x)
# Convolution
x = Conv2D(256, (3, 3), activation="linear", use_bias=False, padding="same")(x)
# Pooling
x = MaxPooling2D(pool_size=(2,2), padding="same")(x)
# Dropout to reduce overfit
x = Dropout(0.3)(x)
# Convolutions
x = Conv2D(512, (3, 3), activation="linear", use_bias=False, padding="same")(x)
# Pooling
x = MaxPooling2D(pool_size=(2,2), padding="same")(x)
# Flatten
x = Flatten()(x)
# Final
x = Dense(512, activation="linear")(x)
# Output layer
preds = Dense(AMOUNT_CLASSES, activation="softmax")(x)
model = Model(inputs=inputs, outputs=preds)

# The below model uses triplet loss, but doesn't work very well

# model = tf.keras.Sequential([
#   Conv2D(32, (3, 3), input_shape=features.shape[1:], activation="relu", use_bias=False, padding="same"),
#   MaxPooling2D(pool_size=(2,2), padding="same"),
#   Dropout(0.3),
#   Conv2D(64, (3, 3), padding="same", activation="relu"),
#   MaxPooling2D(pool_size=(2, 2)),
#   Dropout(0.3),
#   Conv2D(128, (3, 3), input_shape=features.shape[1:], activation="relu", use_bias=False, padding="same"),
#   MaxPooling2D(pool_size=(2,2), padding="same"),
#   Dropout(0.3),
#   Flatten(),
#   Dense(256, activation=None),
#   Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
# ])

model.summary()

# Train model
model.compile(
  loss=tf.keras.losses.sparse_categorical_crossentropy, 
  optimizer=tf.keras.optimizers.Adam(), 
  metrics=[
    # tf.keras.metrics.SparseCategoricalAccuracy(),
    'accuracy'
  ]
)
# model.compile(
#   loss=tfa.losses.TripletSemiHardLoss(),
#   optimizer=tf.keras.optimizers.Adam(),
#   metrics=['accuracy']
# )
model.fit(features_train, labels_train, batch_size=100, validation_split=0.3, epochs=5)

results = model.evaluate(features_test, labels_test, batch_size=100)
print('test loss, test acc:', results)