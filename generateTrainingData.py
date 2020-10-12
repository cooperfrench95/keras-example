import cv2
import os
import random
import numpy as np
import pickle


IMG_SIZE = 64
MIN_PICS_PER_PERSON_IN_TRAINING_SET = 3
TRAIN_DIR = './self-built-masked-face-recognition-dataset/AFDB_face_dataset'
TEST_DIR = './self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset'

# TRAIN_DIR = './lfw_masked/lfw_train'
# TEST_DIR = './lfw_masked/lfw_test'

faces = os.listdir(TRAIN_DIR)
masked = os.listdir(TEST_DIR)

actualMasked = []
for maskedPerson in masked:
    if len(os.listdir(TEST_DIR + '/' + maskedPerson)) > 0:
        actualMasked.append(maskedPerson)
ignorelist = []
for face in faces:
    for maskedPerson in masked:
        if face not in masked:
            ignorelist.append(face)

people_labels = {}
highest_label = 0


def generateData(dir, train=True):

    PEOPLE = os.listdir(dir)
    FEATURES_OUT = 'training_mask_features.pickle'
    LABELS_OUT = 'training_mask_labels.pickle'

    # These will be used during training validation
    FEATURES_TEST_OUT = 'testing_mask_features.pickle'
    LABELS_TEST_OUT = 'testing_mask_labels.pickle'

    # These are the masked images for actual testing of the finished model
    if not train:
        FEATURES_OUT = 'validation_mask_features.pickle'
        LABELS_OUT = 'validation_mask_labels.pickle'

    images = []
    train_images = []

    # Import images
    count = 1
    for person in PEOPLE:
        if person in ignorelist:
            continue
        else:
            count += 1
        path = os.path.join(dir, person)
        amount = len(os.listdir(path))
        # Ensure each person has at least 3 images in the validation set
        amountIgnore = amount - 3
        if amountIgnore < 3:
            continue
        for imageIndex, image in enumerate(os.listdir(path)):
            # Read in images as an array of pixel values
            image_as_numpy_array = cv2.imread(
                os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
            x, y, w, h = cv2.boundingRect(image_as_numpy_array)
            # Crop image, removing the bottom half (hopefully the mouth/mask)
            cropped = image_as_numpy_array[y:int(y+h / 2), x:x+w]
            # Transform image to be e.g. 28x28
            resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
            # Index represents the person's name, because apparently we can't pass a string as a classification type
            if not person in people_labels.keys() and train:
                people_labels[person] = count
            elif not person in people_labels.keys():
                continue
            if train and imageIndex > amountIgnore:
                train_images.append([resized, count])
            elif train and not person in people_labels.keys():
                images.append([resized, people_labels[person]])
            else:
                images.append([resized, people_labels[person]])

    # The images must be shuffled, otherwise the network will learn incorrectly (it will learn to always guess the first guy, then the second guy, etc.)
    random.shuffle(images)
    random.shuffle(train_images)

    features = []
    labels = []

    for f, l in images:
        features.append(f)
        labels.append(l)

    # Transform features into numpy array
    # reshape(amount_of_features, dimension, dimension, 1 = grayscale)
    features = np.array(features).reshape(-1, IMG_SIZE,
                                          IMG_SIZE, 1).astype('float32') / 255.0
    labels = np.array(labels).astype('float32')

    # Save for later
    with open(FEATURES_OUT, 'wb') as f:
        pickle.dump(features, f)

    with open(LABELS_OUT, 'wb') as l:
        pickle.dump(labels, l)

    if train:
        with open(FEATURES_TEST_OUT, 'wb') as f:
            pickle.dump(features, f)

        with open(LABELS_TEST_OUT, 'wb') as l:
            pickle.dump(labels, l)


generateData(TRAIN_DIR)
generateData(TEST_DIR, False)
