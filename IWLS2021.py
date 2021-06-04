import time
import pickle
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score

from Functions.SKlearn_ops import make_sklearn_simple_tree, sklearntree_to_termos
from Functions.CIFAR10_ops import cifar10_class_to_one_hot, unpickle, get_cifar10_ndarray_bits, compact_img, \
    compact_images, load_and_get_binarized, load_and_get_one_binarized
import subprocess
import shlex
import ntpath

from Functions.pla import pla_obj_factory


def compacted_tree_to_original_cifar10_pla(compacted_tree_path, outname):

    compacted_termos = sklearntree_to_termos(compacted_tree_path, 6144)

    for termo in compacted_termos:

        foo_list = ""
        flag = int(termo.get_qt_input() / 2)
        for i in range(flag):
            foo_list = foo_list + "------%s%s" % (termo.get_input()[i*2], termo.get_input()[(i*2)+1])
        termo.set_input(foo_list)
    pla_obj = pla_obj_factory(compacted_termos, outname)
    pla_obj.pla_to_file()


def cifar10_default_decision_tree_pipeline(train_files, train_labels, output_name):
    inputs = 24576
    make_sklearn_simple_tree(train_files, train_labels, output_name)
    termos = sklearntree_to_termos("%s.tree" % output_name, inputs)
    pla_obj = pla_obj_factory(termos, output_name)
    pla_obj.pla_to_file()


def cifar10_custom_decision_tree_pipeline(train_files, train_labels, output_name, pla_inputs):
    inputs = pla_inputs
    make_sklearn_simple_tree(train_files, train_labels, output_name)
    termos = sklearntree_to_termos("%s.tree" % output_name, inputs)
    pla_obj = pla_obj_factory(termos, output_name)
    pla_obj.pla_to_file()


def cifar10_to_pla_one_hot():
    train_files = ["data_batch_1"]
                #"data_batch_2",
                #"data_batch_3",
                #"data_batch_4",
                #"data_batch_5"]

    pla_lines = [".i 24576", ".o 10", ".p 10000"]

    for f in train_files:
        dados = unpickle("CIFAR10/python/%s" % f)
        for counter,v in enumerate(dados.get(b'data')):
            inputs = np.unpackbits(v)
            inputs = list(inputs)
            #print(inputs)
            #print(inputs)
            pla_lines.append("%s %s" % ("".join(["%s" % i for i in inputs]), cifar10_class_to_one_hot(dados.get(b'labels')[counter])))
        print("Finalizou arquivo!!!")

    pla_lines.append(".e")

    with open('data_batch_1.pla', mode='w') as arq:
        for line in pla_lines:
            arq.write(line + "\n")


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

def check_cifar10_datasets_c_and_python():
    fsize = 10000 * (32 * 32 * 3) + 10000
    buffr = np.zeros(fsize, dtype='uint8')

    for i in range(1):
        with open("CIFAR10/bin/data_batch_1.bin", "rb") as bin:
            buffr[i * fsize:(i + 1) * fsize] = np.frombuffer(bin.read(), 'B')
    print(buffr.shape)

    labels = buffr[::3073]
    pixels = np.delete(buffr, np.arange(0, buffr.size, 3073))
    images = pixels.reshape(-1, 3072)

    print(labels.shape)
    print(pixels.shape)
    print(images.shape)

    print(np.unpackbits(images[0][0]))
    print(labels[0:30])

    teste = get_cifar10_ndarray_bits("data_batch_1")

    for counter, img in enumerate(images):
        lista = []
        for i in img:
            t = np.unpackbits(i)
            for y in t:
                lista.append(y)
        new_arr = np.array(lista)

        test = new_arr == teste[0][counter]

        if not test.all():
            print("Tá Mal")


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)

    #saida = aigsim_cifar10_simulation("PLA_dump/AIGs/AIG-Sklearn-Dtree-Cifar10--1-2-3-5-RAW.aig",
    #                                  "PLA_dump/CIFAR10-TRAIN-AND-LABELS/CIFAR10_bin_inputs_data_batch_1.txt",
    #                                  "PLA_dump/teste_aigsim_saida_python.result")

    start = time.time()

    lista = ["CIFAR10/bit-imgs/2-msb-compacted/data_batch_1_binarized",
             "CIFAR10/bit-imgs/2-msb-compacted/data_batch_2_binarized",
             "CIFAR10/bit-imgs/2-msb-compacted/data_batch_3_binarized",
             "CIFAR10/bit-imgs/2-msb-compacted/data_batch_4_binarized",
             "CIFAR10/bit-imgs/2-msb-compacted/data_batch_5_binarized"]

    lista2 = ["CIFAR10/bit-imgs/original/data_batch_1_binarized",
              "CIFAR10/bit-imgs/original/data_batch_2_binarized",
              "CIFAR10/bit-imgs/original/data_batch_3_binarized",
              "CIFAR10/bit-imgs/original/data_batch_4_binarized",
              "CIFAR10/bit-imgs/original/data_batch_5_binarized"]

    dict_train = load_and_get_binarized(lista2)

    clf = RandomForestClassifier(n_estimators=25, n_jobs=-1)

    print('The scikit-learn version is {}.'.format(sklearn.__version__))

    print((np.mean(cross_val_score(clf, dict_train.get(0), dict_train.get(1), cv=10))))

    clf.fit(dict_train.get(0), dict_train.get(1))
    print('Training accuracy: ', clf.score(dict_train.get(0), dict_train.get(1)))

    print(clf.estimators_[0].get_depth())
    print(clf.estimators_[0].get_n_leaves())
    #
    # with open("RandomForest-de1-estimator-0.tree", "w") as arquivo:
    #     arquivo.write(tree.export_text(clf.estimators_[0], max_depth=1000))
    #
    # compacted_tree_to_original_cifar10_pla("RandomForest-de1-estimator-0.tree",
    #                                        "RandomForest-de1-estimator-0-original-size.pla")

    # for counter, rtree in enumerate(clf.estimators_):
    #     with open("RandomForest-estimator-%d.tree" % counter, "w") as arquivo:
    #         arquivo.write(tree.export_text(rtree, max_depth=1000))

    # for i in range(10):
    #     compacted_tree_to_original_cifar10_pla("RandomForest-estimator-%d.tree" % int(i), "RandomForest-estimator-%d-original-size.pla" % int(i))

    print("Tempo: %d" % (time.time() - start))
    time.sleep(60000)

    # cifar10_custom_decision_tree_pipeline(dict_train.get(0), dict_train.get(1), "data_batch_1_2_3_4_5_MEMORY_compacted_dtree", dict_train.get(0).shape[1])

    # compacted_tree_to_original_cifar10_pla("data_batch_1_2_3_4_5_MEMORY_compacted_dtree.tree", "teste_function_compacted_pla")

    # teste = compact_images(dict_train.get(0), 2)
    #
    # print(teste.shape)
    # print(dict_train.get(0).shape)
    #
    # print("Tempo: %d" % (time.time() - start))
    # time.sleep(600)
    #teste = compact_images(train.get(0), 2)
    #print(teste.shape)

    #teste = unpickle("data_batch_1_binarized")
    #print("Tempo: %d" % (time.time()-start))

    #print("Tempo: %d" % (time.time() - start))
    #start = time.time()


    #print("Tempo: %d" % (time.time() - start))
    #print("Foi!! Também")
    #print("Dictionary pickled")
    #time.sleep(33330)

    # Treinar com Decision Tree
    #cifar10_default_decision_tree_pipeline(train.get(0), train.get(1), "data_batch_1_MEMORY_dtree")




    time.sleep(33330)

    train_and_make_decicion_tree()
