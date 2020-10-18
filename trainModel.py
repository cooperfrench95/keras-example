import pickle
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout, Lambda, BatchNormalization
from tensorflow.keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_PICKLE_LABELS = 'training_mask_labels.pickle'
TRAIN_PICKLE_FEATURES = 'training_mask_features.pickle'
TEST_PICKLE_LABELS = 'testing_mask_labels.pickle'
TEST_PICKLE_FEATURES = 'testing_mask_features.pickle'

VALIDATION_PICKLE_FEATURES = 'validation_mask_features.pickle'
VALIDATION_PICKLE_LABELS = 'validation_mask_labels.pickle'

IMG_SIZE = 64
AMOUNT_CLASSES = 64

# Import data
features_train = None
labels_train = None
features_test = None
labels_test = None

features_validate = None
labels_validate = None

with open(TRAIN_PICKLE_FEATURES, 'rb') as f:
    features_train = pickle.load(f)

with open(TRAIN_PICKLE_LABELS, 'rb') as f:
    labels_train = pickle.load(f)

with open(TEST_PICKLE_LABELS, 'rb') as f:
    labels_test = pickle.load(f)

with open(TEST_PICKLE_FEATURES, 'rb') as f:
    features_test = pickle.load(f)

with open(VALIDATION_PICKLE_FEATURES, 'rb') as f:
    features_validate = pickle.load(f)

with open(VALIDATION_PICKLE_LABELS, 'rb') as f:
    labels_validate = pickle.load(f)

# Split into training and validation sets
# features_train = features[:-2000]
# features_test = features[-2000:]
# labels_train = labels[:-2000]
# labels_test = labels[-2000:]


# # Build model
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
# Convolution
x = Conv2D(32, (3, 3), input_shape=features_train.shape[1:],
           activation="relu", use_bias=True, padding="same")(inputs)
# Pooling
x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
# Dropout to reduce overfit
# x = Dropout(0.1)(x)
# Convolution
x = Conv2D(64, (3, 3), activation="linear", use_bias=True, padding="same")(x)
# Pooling
x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
# Dropout to reduce overfit
x = Dropout(0.5)(x)
# # Convolution
# x = Conv2D(64, (3, 3), activation="linear", use_bias=True, padding="same")(x)
# # Pooling
# x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
# # Dropout to reduce overfit
# x = Dropout(0.1)(x)
# # Convolution
# x = Conv2D(128, (3, 3), activation="linear", use_bias=True, padding="same")(x)
# # Pooling
# x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
# # Dropout to reduce overfit
# x = Dropout(0.1)(x)
# Convolutions
x = Conv2D(128, (3, 3), activation="linear",
           use_bias=True, padding="same")(x)
# Pooling
x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
# Flatten
x = Flatten()(x)
# Final
x = Dense(128, activation="linear")(x)
# Output layer
preds = Dense(AMOUNT_CLASSES, activation="softmax")(x)
model = Model(inputs=inputs, outputs=preds)

# The below model uses triplet loss, but doesn't work very well

# model = tf.keras.Sequential([
#   Conv2D(32, (3, 3), input_shape=features.shape[1:], activation="relu", use_bias=True, padding="same"),
#   MaxPooling2D(pool_size=(2,2), padding="same"),
#   Dropout(0.3),
#   Conv2D(64, (3, 3), padding="same", activation="relu"),
#   MaxPooling2D(pool_size=(2, 2)),
#   Dropout(0.3),
#   Conv2D(128, (3, 3), input_shape=features.shape[1:], activation="relu", use_bias=True, padding="same"),
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
        tf.keras.metrics.SparseCategoricalAccuracy(),
        # 'accuracy'
    ]
)
# model.compile(
#   loss=tfa.losses.TripletSemiHardLoss(),
#   optimizer=tf.keras.optimizers.Adam(),
#   metrics=['accuracy']
# )


print(len(features_test), len(features_train), len(features_validate))
print(features_validate[0].shape)
print(features_train.shape)
print(labels_train.shape)

model.fit(features_train, labels_train, batch_size=20, validation_split=0.2,
          epochs=5)


model.save('./model')
del model
model = load_model('./model')
# prediction = model.predict(features_validate, callbacks=[Callbacks()])


results = model.evaluate(
    features_test, labels_test, batch_size=20)
print('test loss, test acc:', results)
