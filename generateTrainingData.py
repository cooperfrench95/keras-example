import cv2
import os
import random
import numpy as np
import pickle
from mtcnn import MTCNN
import matplotlib.pyplot as plt

IMG_SIZE = 64
MIN_PICS_PER_PERSON_IN_TRAINING_SET = 50
TRAIN_DIR = './self-built-masked-face-recognition-dataset/AFDB_face_dataset'
TEST_DIR = './self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset'

# TRAIN_DIR = './lfw_masked/lfw_train'
# TEST_DIR = './lfw_masked/lfw_test'

faces = os.listdir(TRAIN_DIR)
masked = os.listdir(TEST_DIR)

detector = MTCNN()

actualMasked = []
for maskedPerson in faces:
    if (len(actualMasked)) > 63:
        break
    try:
        if len(os.listdir(TEST_DIR + '/' + maskedPerson)) > 0 and len(os.listdir(TRAIN_DIR + '/' + maskedPerson)) > 100:
            actualMasked.append(maskedPerson)
    except:
        continue
ignorelist = []
for face in faces:
    for maskedPerson in masked:
        if face not in actualMasked:
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
    count = -1
    imageCount = 0
    for person in PEOPLE:
        if person in ignorelist:
            continue
        else:
            count += 1
        path = os.path.join(dir, person)
        amount = len(os.listdir(path))
        # Ensure each person has at least 3 images in the validation set
        amountIgnore = amount - MIN_PICS_PER_PERSON_IN_TRAINING_SET
        if amountIgnore < MIN_PICS_PER_PERSON_IN_TRAINING_SET and train:
            continue
        for imageIndex, image in enumerate(os.listdir(path)):
            imageCount += 1
            print('imageCount', imageCount)
            # Read in images as an array of pixel values
            image_as_numpy_array = cv2.imread(
                os.path.join(path, image))
            # x, y, w, h = cv2.boundingRect(image_as_numpy_array)
            cropped = None
            # MTCNN Detects the face
            detected = detector.detect_faces(image_as_numpy_array)
            # Convert back to grayscale
            gray = cv2.cvtColor(image_as_numpy_array, cv2.COLOR_BGR2GRAY)
            if detected and len(detected) > 0:
                # These are the face coordinates
                x, y, w, h = detected[0]['box']
                print(x, y, w, h)
                if y < 0:
                    y = 0
                if x < 0:
                    x = 0
                # Crop to those coordinates
                cropped = gray[y:y+h, x:x+w]
            else:
                cropped = gray
            # Crop image, removing the bottom half (hopefully the mouth/mask)
            # plt.imshow(cropped)
            # plt.show()
            resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
            # cropped = image_as_numpy_array[y:int(y+h / 2), x:x+w]
            # Transform image to be e.g. 28x28
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

    print(people_labels)

    # The images must be shuffled, otherwise the network will learn incorrectly (it will learn to always guess the first guy, then the second guy, etc.)
    random.shuffle(images)
    random.shuffle(train_images)

    features = []
    labels = []

    train_features = []
    train_labels = []

    for f, l in images:
        features.append(f)
        labels.append(l)

    for f, l in train_images:
        train_features.append(f)
        train_labels.append(l)

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
        train_features = np.array(train_features).reshape(-1, IMG_SIZE,
                                                          IMG_SIZE, 1).astype('float32') / 255.0
        train_labels = np.array(train_labels).astype('float32')
        with open(FEATURES_TEST_OUT, 'wb') as f:
            pickle.dump(train_features, f)

        with open(LABELS_TEST_OUT, 'wb') as l:
            pickle.dump(train_labels, l)


generateData(TRAIN_DIR)
generateData(TEST_DIR, False)
