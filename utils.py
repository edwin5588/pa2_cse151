################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import pickle
import numpy as np
import yaml


def write_to_file(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def normalize_z(matrix):
    '''
    Returns normalized matrix
    '''
    mean = np.mean(matrix)
    std = np.std(matrix)

    return (matrix - mean)/std


def load_data():
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')

    return X_train, y_train, X_test, y_test


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    shape = (labels.size, num_classes)
    one_hot = np.zeros(shape)
    rows = np.arange(labels.size)
    one_hot[rows, labels] = 1
    return one_hot


def batch_datatset(x_data, y_labels, b_size):
    '''
    SGD minibatches
    '''
    if x_data.shape[0] != y_labels.shape[0]:
        return Exception('Not the same size!!')

    K = int(y_labels.shape[0]/b_size)
    x_sets = np.empty((K, b_size, x_data.shape[1]))
    y_sets = np.empty((K, b_size, y_labels.shape[1]))

    start = 0
    end = b_size
    for k in range(K):
        # print(start, end)
        x_sets[k, ...] = x_data[start:end, :]
        y_sets[k, ...] = y_labels[start:end, :]
        start = end
        end += b_size

    return x_sets, y_sets

def find_accuracy(predicted, target):
    """
    Find the accuracy given predicted and target

    params:

    predicted : array
    target : array
    """
    return np.sum(predicted == target) / len(target)

def img_2_vec(image):
    '''
    converts a 32x32 image to a 1d vector
    image --> 2d array
    '''
    rows, cols = image.shape[0], image.shape[1]
    img_size = rows * cols
    oned = image.reshape(img_size)
    return oned

def vec_2_img(img_vec):
    '''
    converts 1d vec to 32x32 matrix
    '''
    return img_vec.reshape(32, 32)
