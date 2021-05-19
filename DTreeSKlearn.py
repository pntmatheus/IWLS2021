import os
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus
from IPython.display import Image
from sklearn import tree
from sklearn.preprocessing import Binarizer
from Functions.wekaWrapper import wekatree_to_termos
from Functions.SKlearn_ops import sklearntree_to_termos
from Functions.pla import pla_obj_factory


def read_iop(fn):
    # Reads .i, .o, and .p parameters from the pla benchmark.
    i = o = p = 0
    f = open(fn, 'r')
    for l in f:
        if l.startswith('.i'):
            i = int(l[3:])
        elif l.startswith('.o'):
            o = int(l[3:])
        elif l.startswith('.p'):
            p = int(l[3:])
        elif not l.startswith('.'):
            break
    return i, o, p


def tt2df(fn):
  # Loads a pla bechmark into a Pandas dataframe.
  # The parameter fn is the path of the pla file.
  tt = []
  content = np.genfromtxt(fn, skip_header=3, skip_footer=1, dtype='str')
  for idx in range(0, content.shape[0]):
    tt.append(list(content[idx, 0] + content[idx, 1]))
  return pd.DataFrame(tt)


def draw_dtc(dtc, x, y):
  # This function can be used to visualize the classification tree (if needed).
  # However, very large classification tree take a huge amount or resources.
  dot_data = StringIO()
  export_graphviz(
      dtc,
      out_file=dot_data,
      filled=True,
      rounded=True,
      special_characters=True,
      feature_names=x,
      class_names=y)
  graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
  return (Image(graph.create_png()))


def solucao_casao():
    verilog = open('pla_all_training.v', 'w')

    i, o, p = read_iop("PLA_dump/pla_cifar10_chunk_500_train.pla")
    ts = tt2df("PLA_dump/pla_cifar10_chunk_500_train.pla")
    vs = tt2df("PLA_dump/pla_cifar10_chunk_499_valid.pla")

    print(ts.iloc[:, -10:].values)

    binarizer = Binarizer(threshold=0.0).fit(ts)
    binary_train = binarizer.transform(ts)

    binarizer = Binarizer(threshold=0.0).fit(vs)
    binary_valid = binarizer.transform(vs)

    print(type(ts))
    print(type(binary_train))

    ts = pd.DataFrame(binary_train)
    vs = pd.DataFrame(binary_valid)

    # Split features and the target variable
    X_train = ts.iloc[:, :-10].values
    y_train = ts.iloc[:, -10:].values
    X_val = vs.iloc[:, :-10].values
    y_val = vs.iloc[:, -10:].values

    print(X_train.shape)
    print(y_train.shape)
    print(type(y_train[0]))
    print(type(y_train[0][0]))
    print(y_train[0][0])
    print(y_train[1][0])
    print(y_train[2][0])
    print(y_train[3][0])
    print(y_train)

    # Definition of the classifier
    clf = DecisionTreeClassifier(
        random_state=9856230,
        criterion='gini',
        max_depth=18,
    )

    # Training and validation of the classifier
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_val)
    y_predicted_training = clf.predict(X_train)

    # Generate a Verilog description of the classifier.
    # TODO: create a function out of this code snippet.
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    # print("n_nodes: %s" % str(n_nodes))
    # print("children_left: %s" % str(children_left))
    # print("children_right: %s" % str(children_right))
    # print("feature : %s" % str(feature))
    # print("threshold : %s" % str(threshold))

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    verilog.write('module top (')
    verilog.write('\t' + ', '.join(['x{}'.format(v) for v in np.arange(0, i)]) + ', ')
    # verilog.write('\t' + ', '.join(['y{}'.format(v) for v in np.arange(0, o)]) + 'y);\n')
    verilog.write(', '.join(['y{}'.format(v) for v in np.arange(0, o)]) + ');\n')
    # verilog.write('\t' + 'y);\n')
    verilog.write('input ' + ', '.join(['x{}'.format(v) for v in np.arange(0, i)]) + ';\n')
    verilog.write('output ' + ', '.join(['y{}'.format(v) for v in np.arange(0, o)]) + ';\n')
    # verilog.write('output y;\n')
    verilog.write('wire ' + ', '.join(['n{}'.format(v) for v in np.arange(0, n_nodes)]) + ';\n')
    for i in range(n_nodes):
        if is_leaves[i]:
            verilog.write('assign n{node} = {out_class};\n'.format(
                node=i,
                out_class=np.argmax(clf.tree_.value[i])
            ))
        else:
            verilog.write('assign n{node} = x{feature} ^ 1\'b1 ? n{left} : n{right};\n'.format(
                node=i,
                feature=feature[i],
                left=children_left[i],
                right=children_right[i]
            ))
    verilog.write('assign y = n0;\n')
    verilog.write('endmodule')
    verilog.close()

    with open("tree_test.tree", "w") as arquivo:
        arquivo.write(tree.export_text(clf, max_depth=1000))
    # print(tree.export_text(clf))

def read_file_to_numpy():
    lista = []
    with open("PLA_dump/cifar10_images_binary.txt", "r") as file:
        n = 0
        for line in file.read().splitlines():
            n = n + 1
            print(str(n))
            np_array = np.array(list(line))
            lista.append(np_array.astype(int))
    print(type(lista[0]))
    print(lista[0])
    print(sys.getsizeof(lista))


def teste_gera_sklearn_tree():
    # ts = pd.read_csv("PLA_dump/cifar10_images_binary.txt", header=None, nrows=5000)
    labels = pd.read_csv("PLA_dump/cifar10_images_binary_labels.txt", header=None, nrows=3000)

    tt = []
    content = np.genfromtxt("PLA_dump/cifar10_images_binary.txt", max_rows=3000, dtype=str)
    for idx in range(0, content.shape[0]):
        tt.append(list(content[idx]))
    ts = pd.DataFrame(tt)

    # Definition of the classifier
    clf = DecisionTreeClassifier(
        random_state=9856230,
        criterion='gini',
        max_depth=18,
    )

    clf.fit(ts, labels)

    with open("tree_test2.tree", "w") as arquivo:
        arquivo.write(tree.export_text(clf, max_depth=1000))


def sklearntree_to_termos_OLD(tree_path, qt_inputs):
    ins = qt_inputs

    linha_pla = list()
    control = list()
    termos = list()

    for j in range(ins):
        linha_pla.append("-")

    counter = 0

    with open(tree_path, "r") as tree:
        for linha in tree.read().splitlines():
            original = linha
            # Remove spaces
            linha = linha.replace(" ", "")
            # Remove '-'
            linha = linha.replace("-", "")

            if "class" in linha:
                #out_class = linha.split(":")[0].replace("|", "")
                out_class = linha.split(":")[1]
                print(linha_pla)
                termos.append("%s %s" % ("".join(linha_pla), out_class))
            else:
                # Remove "feature_"
                linha = linha.replace("feature_", "")

                # Pipes count
                pipes = linha.count("|")

                # Pipes clean
                linha = linha.replace("|", "")

                if "<=" in linha:
                    value = '0'
                    attr = int(linha.split("<=")[0])
                else:
                    value = '1'
                    attr = int(linha.split(">")[0])

                if counter < pipes:
                    control.append(attr)
                    counter = counter + 1
                else:
                    if counter > pipes:
                        for i in range(counter - pipes):
                            print(control[-1])
                            print(original)
                            linha_pla[control[-1]] = "-"
                            control.pop(-1)
                        counter = pipes

                linha_pla[attr] = value

        return termos


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    np.set_printoptions(threshold=sys.maxsize)

    termos = sklearntree_to_termos("Sklearn-Dtree-Cifar10--1-2-3-5.tree", 24576)
    pla = pla_obj_factory(termos, "Sklearn-Dtree-Cifar10--1-2-3-5")
    print(pla.get_nome())
    print(pla.get_qt_inputs())
    print(pla.get_qt_outputs())
    print(len(pla.get_termos()))
    print(pla.get_type())

    pla.pla_to_file()

