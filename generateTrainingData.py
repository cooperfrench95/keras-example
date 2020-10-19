import cv2
import os
import random
import pickle
import numpy as np
import tensorflow as tf
###############################
##################################


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

IMG_SIZE = 64
ImgSourceDirectory = './self-built-masked-face-recognition-dataset/AFDB_face_dataset'


#ImgSourceDirectory = './Indian-celebritiesBase/Indian-celebrities'
# https://gist.github.com/SainaGhosh
# poor results on any model using this dataset - no idea why

#ImgSourceDirectory = './CASIA-WebFace/CASIA-WebFace'
# https://drive.google.com/open?id=1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz
# The CASIA dataset is annotated with 10,575 unique people with 494,414 images in total.


#ImgSourceDirectory = './self-built-masked-face-recognition-dataset/AFDB_face_dataset'
# Default dataset


PEOPLE = os.listdir(ImgSourceDirectory)

images = []

# Import images
for index, person in enumerate(PEOPLE):
    path = os.path.join(ImgSourceDirectory, person)
    # print(path)
    for image in os.listdir(path):
        # Read in images as an array of pixel values
        image_as_numpy_array = cv2.imread(
            os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
        print(path, image)


# The images must be shuffled, otherwise the network will learn incorrectly (it will learn to always guess the first guy, then the second guy, etc.)
random.shuffle(images)

features = []
labels = []

for f, l in images:
    features.append(f)
    labels.append(l)

# Transform features into numpy array
# reshape(amount_of_features, dimension, dimension, 1 = grayscale)
features = np.array(features).reshape(-1, IMG_SIZE,
                                      IMG_SIZE, 1).astype('float32') / 255.0
labels = np.array(labels).astype('float64')

# Save for later
with open('features.pickle', 'wb') as f:
    pickle.dump(features, f)


with open('labels.pickle', 'wb') as l:
    pickle.dump(labels, l)
