import numpy as np
import time
import gc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from IWLS2021 import unpickle, cifar10_default_decision_tree_pipeline


def add_train_batch(b_dict, b_number=None):
    train_files = {1: "data_batch_1",
                   2: "data_batch_2",
                   3: "data_batch_3",
                   4: "data_batch_4",
                   5: "data_batch_5"}

    inputs = b_dict.get(0)
    labels = b_dict.get(1)

    if b_number is None:
        for train in train_files:
            dados = unpickle("CIFAR10/python/%s" % train_files.get(train))
            for v in dados.get(b'data'):
                t = np.unpackbits(v)
                inputs.append(t)
            for lab in dados.get(b'labels'):
                labels.append(lab)
    else:
        dados = unpickle("CIFAR10/python/%s" % train_files.get(b_number))
        for v in dados.get(b'data'):
            t = np.unpackbits(v)
            inputs.append(t)
        for lab in dados.get(b'labels'):
            labels.append(lab)

    b_dict[0] = inputs
    b_dict[1] = labels

    return b_dict


if __name__ == "__main__":

    files = ["data_batch_1_binarized",
             "data_batch_2_binarized",
             "data_batch_3_binarized",
             "data_batch_4_binarized",
             "data_batch_5_binarized"]
    dict_list = []
    labels = []
    for file in files:
        dictionary = unpickle("CIFAR10/bit-imgs/original/%s" % file)
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

    print(train.shape)
    print(labs.shape)

    start = time.time()
    cifar10_default_decision_tree_pipeline(train, labs, "data_batch_1_2_3_4_5_MEMORY_dtree")
    print("Tempo: %d" % (time.time() - start))

    print("Pronto!!!")
    time.sleep(5000)
    for j in range(1, 30):
        train_dict = {0: [],
                      1: []}
        start_time = time.time()

        for i in range(j):
            train_dict = add_train_batch(train_dict)

        train_inputs = np.array(train_dict.get(0))
        train_labels = np.array(train_dict.get(1))

        clf = RandomForestClassifier(n_estimators=20, n_jobs=-1, min_samples_leaf=8)
        clf.fit(train_inputs, train_labels)

        print("RandomForest with %d CIFAR10 training in %f " % (train_inputs.shape[0], (time.time() - start_time)))
        gc.collect()
