from __future__ import annotations
import pickle
import time

from multipledispatch import dispatch
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


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


@dispatch(str)
def get_cifar10_ndarray_bits(cifar10_file):
    inputs = []
    labels = []

    file = unpickle("CIFAR10/python/%s" % cifar10_file)
    for img in file.get(b'data'):
        #img_input = np.zeros(24576, dtype='int8')
        img_input = []
        init_flag = 0
        for pixel_channel in img:
            t = np.unpackbits(pixel_channel, bitorder='little')
            # t_inv = t[::-1]
            img_input.append(t)
            #for c, v in enumerate(t):
            #    img_input[init_flag+c] = v
            #init_flag = init_flag + 8
        narray = np.array(img_input, dtype='int8')
        narray = narray.flatten()

        inputs.append(narray)

    for lab in file.get(b'labels'):
        labels.append(lab)

    #train = np.array(inputs)
    #lista = None
    #labs = np.array(labels)
    #labels = None

    dictionary = {0: inputs,
                  1: labels}

    return dictionary


@dispatch(list)
def get_cifar10_ndarray_bits(cifar10_files):
    lista = []
    labels = []
    for train in cifar10_files:
        dados = unpickle("CIFAR10/python/%s" % train)
        for v in dados.get(b'data'):
            t = np.unpackbits(v)
            lista.append(t)

        for lab in dados.get(b'labels'):
            labels.append(lab)

    #train = np.array(lista)
    #lista = None
    #labs = np.array(labels)
    #labels = None

    dictionary = {0: lista,
                  1: labels}

    return dictionary


if __name__ == "__main__":
    #np.set_printoptions(threshold=np.inf)
    start = time.time()
    print("Tempo = %d" % (time.time() - start))
    dataset_dict = get_cifar10_ndarray_bits("data_batch_1")
    print("Tempo = %d" % (time.time()-start))
    print("Montou o dictionary!!!!")
    pla_lines = [".i 24576", ".o 10", ".p 10000"]
    for counter,input in enumerate(dataset_dict.get(0)):
        pla_lines.append("%s %s" % ("".join(["%s" % i for i in input]), cifar10_class_to_one_hot(dataset_dict.get(1)[counter])))

    pla_lines.append(".e")

    print("Montou o vetor!!!!!!!")
    #time.sleep(36000)
    with open('data_batch_1_ALL_INVERTED.pla', mode='w') as arq:
        for line in pla_lines:
            arq.write(line + "\n")
    print("Foi!!!!!!!!!!!")
