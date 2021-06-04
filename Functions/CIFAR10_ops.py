from __future__ import annotations

import gc
import pickle
import time

from multipledispatch import dispatch
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def load_and_get_one_binarized(file):
    dictionary = unpickle("%s" % file)
    return dictionary

# no mínimo 2 datasets!!!
def load_and_get_binarized(file_list):
    dict_list = []
    labels = []
    for file in file_list:
        dictionary = unpickle("%s" % file)
        dict_list.append(dictionary.get(0))
        labels.append(dictionary.get(1))

    dictionary = None

    train = dict_list[0]
    labs = labels[0]

    for i in dict_list[1:]:
        train = np.vstack((train, i))

    for i in labels[1:]:
        labs = np.concatenate((labs, i))

    dict_list = None
    labels = None
    gc.collect()

    return {0: train, 1: labs}


def cifar10_class_to_one_hot(int_class):
    # For ABC "iwls21test" the bits orders was little endian
    classes = {0: "1000000000",  # airplane
               1: "0100000000",  # automobile
               2: "0010000000",  # bird
               3: "0001000000",  # cat
               4: "0000100000",  # deer
               5: "0000010000",  # dog
               6: "0000001000",  # frog
               7: "0000000100",  # horse
               8: "0000000010",  # ship
               9: "0000000001"}  # truck
    return classes.get(int_class)


def compact_images(images, msb):
    c_imgs = []
    for i in range(images.shape[0]):
        c_imgs.append(compact_img(images[i], msb))
    return np.array(c_imgs, 'int8')


def compact_img(array_bits, msb):
    img = array_bits.reshape(1, 3072, 8)
    new_array = []

    for i in img[0]:
        new_array.append(i[-msb:])

    return np.array(new_array, dtype='int8').flatten()


@dispatch(str)
def get_cifar10_ndarray_bits(cifar10_file):
    inputs = []
    labels = []

    file = unpickle("CIFAR10/python/%s" % cifar10_file)
    for img in file.get(b'data'):
        img_input = []
        for pixel_channel in img:
            t = np.unpackbits(pixel_channel, bitorder='little')
            # t_inv = t[::-1]
            img_input.append(t)
        narray = np.array(img_input, dtype='int8')
        narray = narray.flatten()
        inputs.append(narray)

    for lab in file.get(b'labels'):
        labels.append(lab)

    inputs = np.array(inputs, dtype='int8')
    labels = np.array(labels, dtype='int8')

    dictionary = {0: inputs,
                  1: labels}

    return dictionary


@dispatch(list)
def get_cifar10_ndarray_bits(cifar10_files):
    inputs = []
    labels = []
    for train in cifar10_files:
        file = unpickle("CIFAR10/python/%s" % train)
        for img in file.get(b'data'):
            img_input = []
            for pixel_channel in img:
                t = np.unpackbits(pixel_channel, bitorder='little')
                img_input.append(t)
            narray = np.array(img_input, dtype='int8')
            narray = narray.flatten()
            inputs.append(narray)

        for lab in file.get(b'labels'):
            labels.append(lab)

    labs = np.array(labels, dtype='int8')
    labels = None

    dictionary = {0: inputs,
                  1: labs}

    return dictionary
