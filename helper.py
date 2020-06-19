import pickle
import numpy as np

def pickle_this(obj_to_pickle, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj_to_pickle, file)

def unpickle_this(filename):
    with open(filename, 'rb') as file:
        unpickled = pickle.load(file)
    return unpickled