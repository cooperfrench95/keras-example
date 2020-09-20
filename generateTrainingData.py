import cv2
import os
import random
import numpy as np
import pickle


IMG_SIZE = 64
NO_MASK_DIR = './self-built-masked-face-recognition-dataset/AFDB_face_dataset'
MASK_DIR = './self-built-masked-face-recognition-dataset/AFDB_face_dataset'

PEOPLE = os.listdir(NO_MASK_DIR)


images = []

# Import images
for index, person in enumerate(PEOPLE):
  path = os.path.join(NO_MASK_DIR, person)
  for image in os.listdir(path):
    # Read in images as an array of pixel values
    image_as_numpy_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
    # Transform image to be e.g. 28x28
    resized = cv2.resize(image_as_numpy_array, (IMG_SIZE, IMG_SIZE))
    # Index represents the person's name, because apparently we can't pass a string as a classification type
    images.append([resized, index])

# The images must be shuffled, otherwise the network will learn incorrectly (it will learn to always guess the first guy, then the second guy, etc.)
random.shuffle(images)

features = []
labels = []

for f, l in images:
  features.append(f)
  labels.append(l)

# Transform features into numpy array
# reshape(amount_of_features, dimension, dimension, 1 = grayscale)
features = np.array(features).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
labels = np.array(labels).astype('float32')

# Save for later
with open('training_no_mask_features.pickle', 'wb') as f:
  pickle.dump(features, f)

with open('training_no_mask_labels.pickle', 'wb') as l:
  pickle.dump(labels, l)

