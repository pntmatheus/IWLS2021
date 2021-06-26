import os
import time
import pickle
import numpy as np
import shutil
from sklearn.cluster import KMeans

from Functions.scriptABCFunctions import gera_abc_aig
from Functions.SKlearn_ops import make_sklearn_simple_tree, sklearntree_to_termos, sklearntree_to_pla
from Functions.SKlearn_ops import rf_train_and_export
from Functions.CIFAR10_ops import cifar10_class_to_one_hot, unpickle, get_cifar10_ndarray_bits, compact_img, \
    compact_images, load_and_get_binarized, load_and_get_one_binarized
import subprocess
import shlex
import ntpath

from Functions.pla import pla_obj_factory

cifar10_integers = ["CIFAR10/bit-imgs/integer/data_batch_1",
                    "CIFAR10/bit-imgs/integer/data_batch_2",
                    "CIFAR10/bit-imgs/integer/data_batch_3",
                    "CIFAR10/bit-imgs/integer/data_batch_4",
                    "CIFAR10/bit-imgs/integer/data_batch_5"]

data_batches = ["CIFAR10/bit-imgs/2-msb-compacted/data_batch_1_binarized_2msb",
                "CIFAR10/bit-imgs/2-msb-compacted/data_batch_2_binarized_2msb",
                "CIFAR10/bit-imgs/2-msb-compacted/data_batch_3_binarized_2msb",
                "CIFAR10/bit-imgs/2-msb-compacted/data_batch_4_binarized_2msb",
                "CIFAR10/bit-imgs/2-msb-compacted/data_batch_5_binarized_2msb"]


def iwls21_team_pipeline(dataset, max_depth, n_estimators, rf_name, out_folder, remove_pla=False):
    tree_folder = "%s/tree" % out_folder
    aig_folder = "%s/aig" % out_folder

    try:
        os.mkdir(out_folder)
        print("pasta small2 ok")
    except FileExistsError:
        print("pasta small2 jah existia")
        pass

    try:
        os.mkdir(tree_folder)
        print("pasta tree ok")
    except FileExistsError:
        print("pasta tree ja existia")
        pass

    try:
        os.mkdir(aig_folder)
        print("pasta aig ok")
    except FileExistsError:
        print("pasta aig ja existia")
        pass

    # train and export RF model
    rf_train_and_export(dataset, max_depth, n_estimators, rf_name, tree_folder)

    # Convert trees to aig
    all_trees_to_aig(tree_folder, aig_folder, remove_pla=remove_pla)

    print("OK!!!")


def all_plas_to_aig(folder, aig_folder):
    pla_files = [x for x in os.scandir(folder)]
    #print(pla_files[0].path)
    #print(type(pla_files[0]))
    for pla in pla_files:
        pla_to_aig(pla, aig_folder)


def all_trees_to_aig(tree_folder, aig_folder, compacted=False, remove_pla=False):
    tree_files = [x for x in os.scandir(tree_folder)]
    trees_path = []
    for i in tree_files:
        trees_path.append(i.path)

    # make pla folder
    try:
        os.mkdir("%s/pla" % aig_folder)
    except FileExistsError:
        pass

    compacted_trees_to_original_cifar10_pla(trees_path, "%s/pla" % aig_folder)
    all_plas_to_aig("%s/pla" % aig_folder, aig_folder)
    if remove_pla:
        shutil.rmtree("%s/pla" % aig_folder)

def pla_to_aig(pla_osdirentry, out_folder):
    filename = pla_osdirentry.name.replace(".pla", "")
    cmds = ["resyn2", "resyn2", "resyn2", "resyn2", "resyn3", "resyn3", "resyn3", "resyn2", "resyn2"]
    print(gera_abc_aig("\'%s\'" % pla_osdirentry.path, "\'%s/%s.aig\'" % (out_folder, filename), cmds))


def func_tree_to_aig():
    path = "PLA_dump/PRELIMINAR_SOLUTIONS/Large/RandomForest_AND_Voter/full-bit"
    cmds = ["resyn2", "resyn2", "resyn2", "resyn2", "resyn3", "resyn3", "resyn3", "resyn2", "resyn2"]

    filename = "RandomForest-binarized-FULL_inputs-0-20"

    sklearntree_to_pla("%s/tree/%s.tree" % (path, filename), 24576, "%s/pla/%s" % (path, filename))
    print(gera_abc_aig("\'%s/pla/%s.pla\'" % (path, filename), "\'%s/aig/%s.aig\'" % (path, filename), cmds))

#def cifar10_default_inputs():
#    return unpickle("DEFAULT-CIFAR10-VARIABLES.pickled")


#def cifar10_default_outputs(tree_id=None):

#    if tree_id is not None:
#        values = []
#        for i in range(10):
#            values.append("C%dt%d" % (i, tree_id))
#        return values
#    else:
#        return ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]


#def make_pla_variables_labels(type, tree_cifar_id=None):
#    switcher = {
#        "cifar10_default_inputs": cifar10_default_inputs(),
#        "cifar10_default_outputs": cifar10_default_outputs(tree_id=tree_cifar_id)
#    }

#    return switcher.get(type)


def compacted_trees_to_original_cifar10_pla(compacted_tree_paths, out_folder):
    out_names = []
    for counter, tree in enumerate(compacted_tree_paths):
        tree_name = ntpath.basename(tree)
        print(tree_name)
        out_name = "%s/%s" % (out_folder, tree_name)
        print(out_name)
        compacted_tree_to_original_cifar10_pla(tree, out_name)
        out_names.append(out_name + ".pla")
    return out_names


def compacted_tree_to_original_cifar10_pla(compacted_tree_path, outname, in_labels=None, out_labels=None):

    compacted_termos = sklearntree_to_termos(compacted_tree_path, 6144)
    for termo in compacted_termos:

        foo_list = ""
        flag = int(termo.get_qt_input() / 2)
        for i in range(flag):
            foo_list = foo_list + "------%s%s" % (termo.get_input()[i*2], termo.get_input()[(i*2)+1])
        termo.set_input(foo_list)
    pla_obj = pla_obj_factory(compacted_termos, outname)
    pla_obj.pla_to_file(in_labels=in_labels, out_labels=out_labels)


#def cifar10_default_decision_tree_pipeline(train_files, train_labels, output_name):
#    inputs = 24576
#    make_sklearn_simple_tree(train_files, train_labels, output_name)
#    termos = sklearntree_to_termos("%s.tree" % output_name, inputs)
#    pla_obj = pla_obj_factory(termos, output_name)
#    pla_obj.pla_to_file()


#def cifar10_custom_decision_tree_pipeline(train_files, train_labels, output_name, pla_inputs):
#    inputs = pla_inputs
#    make_sklearn_simple_tree(train_files, train_labels, output_name)
#    termos = sklearntree_to_termos("%s.tree" % output_name, inputs)
#    pla_obj = pla_obj_factory(termos, output_name)
#   pla_obj.pla_to_file()


#def cifar10_to_pla_one_hot():
#    train_files = ["data_batch_1"]
#                #"data_batch_2",
#                #"data_batch_3",
                #"data_batch_4",
                #"data_batch_5"]

#    pla_lines = [".i 24576", ".o 10", ".p 10000"]

#    for f in train_files:
#        dados = unpickle("CIFAR10/python/%s" % f)
#        for counter,v in enumerate(dados.get(b'data')):
#            inputs = np.unpackbits(v)
#            inputs = list(inputs)
#            #print(inputs)
#            #print(inputs)
#           pla_lines.append("%s %s" % ("".join(["%s" % i for i in inputs]), cifar10_class_to_one_hot(dados.get(b'labels')[counter])))
#        print("Finalizou arquivo!!!")

#    pla_lines.append(".e")

#    with open('data_batch_1.pla', mode='w') as arq:
#        for line in pla_lines:
#            arq.write(line + "\n")

def aigsim_cifar10_simulation(aig_path, inputs_path, output_path):
    command = "./tools/aigsim %s %s --supress" % (aig_path, inputs_path)
    print(command)
    aigsim_cmd = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    # O retorno desta funcao sera [0] = stdout e [1] = sterr (erro)
    exec_aigsim = aigsim_cmd.communicate()

    with open(output_path, "w") as file:
        file.write(exec_aigsim[0].decode())

    return exec_aigsim


def create_simulation_files_for_cifar10(aig_path) -> list:
    cifar10_train_inputs = ["PLA_dump/CIFAR10-TRAIN-AND-LABELS/CIFAR10_bin_inputs_data_batch_1.txt",
                            "PLA_dump/CIFAR10-TRAIN-AND-LABELS/CIFAR10_bin_inputs_data_batch_2.txt",
                            "PLA_dump/CIFAR10-TRAIN-AND-LABELS/CIFAR10_bin_inputs_data_batch_3.txt",
                            "PLA_dump/CIFAR10-TRAIN-AND-LABELS/CIFAR10_bin_inputs_data_batch_4.txt",
                            "PLA_dump/CIFAR10-TRAIN-AND-LABELS/CIFAR10_bin_inputs_data_batch_5.txt"]

    result_files_list = []
    b_name = ntpath.basename(aig_path).replace(".aig", "")
    for train in cifar10_train_inputs:
        cifar10_b_name = ntpath.basename(train).replace("CIFAR10_bin_inputs", "")
        cifar10_b_name = cifar10_b_name.replace(".txt", "")
        result_path = "PLA_dump/AIGs/%s%s.result" % (b_name, cifar10_b_name)
        aigsim_cifar10_simulation(aig_path, train, result_path)
        result_files_list.append(result_path)

    result_files_list = sorted(result_files_list)

    return result_files_list


def aig_cifar10_accuracy(aig_simulation_path, cifar10_label_path):
    gab_list = []
    resp_list = []

    with open(aig_simulation_path, "r") as r_file:
        for line in r_file:
            resp_list.append(line[:-1])

    with open(cifar10_label_path) as g_file:
        for line in g_file:
            gab_list.append(line[:-1])
    flag = 0
    for c, e in enumerate(resp_list):
        one_hot = cifar10_class_to_one_hot(int(gab_list[c]))
        resp = e.split(" ")[2]
        if one_hot == resp:
            flag += 1
    print(flag)


def make_3x3_cifar10_filter():
    cifar10 = load_and_get_binarized(cifar10_integers)

    vetor_final = []
    for img in cifar10.get(0):
        int_img = np.reshape(img, (3, 1024))
        int_img = np.reshape(int_img, 3072, order='F')
        int_img = np.reshape(int_img, (32, 32, 3))

        # Here the range jump is the "filter stride"
        # And x/y range is the filter size
        for i in range(0, int_img.shape[0], 2):
            for j in range(0, int_img.shape[1], 2):
                new_input = []
                if i < 30 and j < 30:
                    for x in range(3):
                        for y in range(3):
                            new_input.append(int_img[i + x][j + y])

                    # print(vetor)
                else:
                    for x in range(3):
                        for y in range(3):
                            X = i + x
                            Y = j + y

                            if X > 31 or Y > 31:
                                new_input.append(np.array([0, 0, 0], dtype='uint8'))
                            else:
                                new_input.append(int_img[X][Y])
                vetor = np.array(new_input).flatten()
                vetor_final.append(vetor)

    new_array = np.array(vetor_final)
    print(new_array.shape)
    with open("data_batch_1_2_3_4_5-3x3filter-without-labels", "wb") as file:
        pickle.dump(new_array, file)


def train_kmeans_and_pickle(k, dataset, pickle_name):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dataset)
    with open(pickle_name, "wb") as file:
        pickle.dump(kmeans, file)


def get_cifar10_3x3_filter(img):
    filter3x3 = []
    int_img = np.reshape(img, (3, 1024))
    int_img = np.reshape(int_img, 3072, order='F')
    int_img = np.reshape(int_img, (32, 32, 3))

    # Here the range jump is the "filter stride"
    # And x/y range is the filter size
    for i in range(0, int_img.shape[0], 2):
        for j in range(0, int_img.shape[1], 2):
            new_input = []
            if i < 30 and j < 30:
                for x in range(3):
                    for y in range(3):
                        new_input.append(int_img[i + x][j + y])

                # print(vetor)
            else:
                for x in range(3):
                    for y in range(3):
                        X = i + x
                        Y = j + y

                        if X > 31 or Y > 31:
                            new_input.append(np.array([0, 0, 0], dtype='uint8'))
                        else:
                            new_input.append(int_img[X][Y])

            vetor = np.array(new_input).flatten()
            filter3x3.append(vetor)
    return filter3x3


def cifar10_custom_filter(img, start_row, start_column, width, height):

    filtered_img = []

    img_int = np.reshape(img, (3, 1024))
    img_int = np.reshape(img_int, 3072, order='F')
    img_int = np.reshape(img_int, (32, 32, 3))

    sqr_width = start_column+width
    sqr_height = start_row+height

    if sqr_width < img_int.shape[0]+1 and sqr_height < img_int.shape[1]+1:
        for x in range(start_row, sqr_height):
            for y in range(start_column, sqr_width):
                filtered_img.append(img_int[x][y])
    else:
        print("Filter out of size!!")
        return None

    f_img = np.array(filtered_img).flatten()

    return f_img


def cifar_kmeans_predict():
    kmeans_model: KMeans = unpickle("KMEANS-K6-FULL-CIFAR10")
    cifar10 = load_and_get_binarized(cifar10_integers)
    new_inputs = []

    for img in cifar10.get(0):
        to_pred = get_cifar10_3x3_filter(img)
        new_inputs.append(kmeans_model.predict(to_pred))

    nd_array = np.array(new_inputs)

    new_dict = {0: nd_array, 1: cifar10.get(1)}
    with open("new_data_batch_1_2_3_4_5_KMEANS6", "wb") as file:
        pickle.dump(new_dict, file)


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)

    start = time.time()

    # ['RF_Tree_individual_name (str)', 'output_folder (str)', 'RF_max_depth (int)', 'n_estimators (int)']
    solutions = [["RF10-depth08-pipelined", "PLA_dump/PRELIMINAR_SOLUTIONS/small-10-depth08", 10, 8],
                 ["RF50-depth09-pipelined", "PLA_dump/PRELIMINAR_SOLUTIONS/medium-50-depth09", 50, 9],
                 ["RF130-depth11-pipelined", "PLA_dump/PRELIMINAR_SOLUTIONS/large-130-depth11", 130, 11]]

    dataset = load_and_get_binarized(data_batches)

    for solution in solutions:
        iwls21_team_pipeline(dataset, solution[3], solution[2], solution[0], solution[1], remove_pla=True)