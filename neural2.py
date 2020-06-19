import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle

DATADIR = '/Users/anthonymenninger/Documents/Code/PythonCode/Neural/PetImages'
CATEGORIES = ['Dog', 'Cat']
IMG_SIZE = 100

training_data = []


def pickle_this(obj_to_pickle, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj_to_pickle, file)

def unpickle_this(filename):
    with open(filename, 'rb') as file:
        unpickled = pickle.load(file)
    return unpickled

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

    random.shuffle(training_data)

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    pickle_this(X, 'X.pickle')
    pickle_this(y, 'y.pickle')

    return X, y

X, y = create_training_data()


