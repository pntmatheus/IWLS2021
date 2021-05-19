import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import Functions.pla
from Functions.SKlearn_ops import cifar10_class_to_one_hot
import pickle
import subprocess
import shlex
import ntpath

def pla_obj_to_arff(pla, gera_arquivo=False, unique=False, path="tmp_iwls2020/ARFF"):
    # Retira os repetidos
    if unique:
        pla.turn_termos_unique()

    conteudo = list()
    # Adicionar primeira linha do ARFF
    conteudo.append("@relation %s" % pla.get_nome().replace(".pla", ""))

    # Adicionar os atributos de entrada no ARFF
    for i in range(pla.get_qt_inputs()):
        conteudo.append("@attribute a%d {0,1}" % (i))

    # Adicionar os atributos de saida no ARFF
    #for i in range(pla.get_qt_outputs()):
    conteudo.append("@attribute output {0000000001, 0000000010, 0000000100, 0000001000, 0000010000, 0000100000, "
                    "0001000000, 0010000000, 0100000000, 1000000000}")

    # Adicionar o dataset
    conteudo.append("@data")
    for t in pla.get_termos():
        termo_format = "%s%s" % (
            #"".join(["%s," % i for i in t.get_input()]), "".join(["%s" % i for i in t.get_output()]))
            "".join(["%s," % i for i in t.get_input()]), "%s" % t.get_output())
        conteudo.append(termo_format)

    if gera_arquivo:
        with open("%s.arff" % (pla.get_nome()), "w") as arquivo:
            for i in conteudo:
                arquivo.write(i + "\n")


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def make_sklearn_simple_tree(train, labels, tree_out_name):
    # Definition of the classifier
    clf = DecisionTreeClassifier(
        random_state=9856230,
        criterion='gini',
        max_depth=50,
        ccp_alpha=0.015
    )

    clf.fit(train, labels)

    with open("%s.tree" % tree_out_name, "w") as arquivo:
        arquivo.write(tree.export_text(clf, max_depth=1000))


def train_and_make_decicion_tree():
    train_files = ["data_batch_1",
                   "data_batch_2",
                   "data_batch_3",
                   # "data_batch_4",
                   "data_batch_5"]

    lista = []
    labels = []
    for train in train_files:
        dados = unpickle("CIFAR10/python/%s" % train)
        for v in dados.get(b'data'):
            t = np.unpackbits(v)
            lista.append(t)

        for lab in dados.get(b'labels'):
            labels.append(lab)

    train = np.array(lista)
    lista = None
    labs = np.array(labels)
    labels = None

    print(train.shape)
    print(labs.shape)

    make_sklearn_simple_tree(train, labs, "Matheus_teste_ccp")


def create_cifar10_training_and_labels_files():
    train_files = ["data_batch_1",
                   "data_batch_2",
                   "data_batch_3",
                   "data_batch_4",
                   "data_batch_5"]

    lista = []
    labels = []

    dados = unpickle("CIFAR10/python/%s" % train_files[0])
    print(dados.get(b'data')[0])
    teste = np.unpackbits(dados.get(b'data')[0])
    print(teste)
    print(teste.dtype)
    print(teste.tostring())
    print(teste.tostring().decode('ascii'))

    retorno = "".join(map(str, teste))
    print(retorno)
    print(len(retorno))

    for train in train_files:
        train_files = []
        train_labels = []
        dados = unpickle("CIFAR10/python/%s" % train)

        y = 0
        for img in dados.get(b'data'):
            img_bin = np.unpackbits(img)
            train_files.append("".join(map(str, img_bin)))

        for lab in dados.get(b'labels'):
            train_labels.append(lab)

        with open("PLA_dump/CIFAR10_bin_inputs_%s.txt" % train, "w") as file:
            last = len(train_files)
            for counter, t in enumerate(train_files, start=1):
                if counter == last:
                    file.write(t)
                else:
                    file.write(t + "\n")
            print("Acabou o arquivo CIFAR10_bin_inputs_%s.txt" % train)

        with open("PLA_dump/CIFAR10_bin_labels_%s.txt" % train, "w") as file:
            last = len(train_labels)
            for counter, t in enumerate(train_labels, start=1):
                if counter == last:
                    file.write(str(t))
                else:
                    file.write(str(t) + "\n")

        print("Acabou o arquivo CIFAR10_bin_labels_%s.txt" % train)


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

    # Bater o gabrito
    flag = 0
    for c, e in enumerate(resp_list):
        one_hot = cifar10_class_to_one_hot(int(gab_list[c]))
        resp = e.split(" ")[2]
        if one_hot == resp:
            flag += 1
    print(flag)


if __name__ == "__main__":
    gabaritos = ["PLA_dump/CIFAR10-TRAIN-AND-LABELS/CIFAR10_bin_labels_data_batch_1.txt",
                 "PLA_dump/CIFAR10-TRAIN-AND-LABELS/CIFAR10_bin_labels_data_batch_2.txt",
                 "PLA_dump/CIFAR10-TRAIN-AND-LABELS/CIFAR10_bin_labels_data_batch_3.txt",
                 "PLA_dump/CIFAR10-TRAIN-AND-LABELS/CIFAR10_bin_labels_data_batch_4.txt",
                 "PLA_dump/CIFAR10-TRAIN-AND-LABELS/CIFAR10_bin_labels_data_batch_5.txt"]

    #respostas = ["PLA_dump/AIGs/AIG-Sklearn-Dtree-Cifar10--1-2-3-5-resyn2-x1_data_batch_1.result",
    #             "PLA_dump/AIGs/AIG-Sklearn-Dtree-Cifar10--1-2-3-5-resyn2-x1_data_batch_2.result",
    #             "PLA_dump/AIGs/AIG-Sklearn-Dtree-Cifar10--1-2-3-5-resyn2-x1_data_batch_3.result",
    #             "PLA_dump/AIGs/AIG-Sklearn-Dtree-Cifar10--1-2-3-5-resyn2-x1_data_batch_4.result",
    #             "PLA_dump/AIGs/AIG-Sklearn-Dtree-Cifar10--1-2-3-5-resyn2-x1_data_batch_5.result"]

    #saida = aigsim_cifar10_simulation("PLA_dump/AIGs/AIG-Sklearn-Dtree-Cifar10--1-2-3-5-RAW.aig",
    #                                  "PLA_dump/CIFAR10-TRAIN-AND-LABELS/CIFAR10_bin_inputs_data_batch_1.txt",
    #                                  "PLA_dump/teste_aigsim_saida_python.result")

    time.sleep(30)
    train_and_make_decicion_tree()

    respostas = create_simulation_files_for_cifar10("PLA_dump/AIGs/AIG-Sklearn-Dtree-Cifar10--1-2-3-5-RAW.aig")
    print(respostas)

    for counter, r in enumerate(respostas):
        aig_cifar10_accuracy(r, gabaritos[counter])
