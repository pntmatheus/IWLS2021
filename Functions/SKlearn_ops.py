import os

from Functions.termo import Termo
from Functions.pla import pla_obj_factory
from Functions.CIFAR10_ops import cifar10_class_to_one_hot
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


def rf_train_and_export(dataset, max_depth, n_estimators, rf_name, out_folder):
    clf = random_forest_train(dataset, max_depth, n_estimators=n_estimators)
    print("RF Trained!!!")
    export_random_forest_trees(clf, rf_name, out_folder)
    print("%s exported at %s folder!!!" % (rf_name, out_folder))


def export_random_forest_trees(rf_model, rf_name, out_folder):
    # make folder if dows not exist
    try:
        os.mkdir(out_folder)
    except FileExistsError:
        pass

    for counter, estimator in enumerate(rf_model.estimators_):
        with open("%s/%s--%d-%d.tree" % (out_folder, rf_name, counter, (len(rf_model) - 1)), "w") as file:
            file.write(tree.export_text(estimator, max_depth=1000))




def random_forest_train(dataset, max_depth, n_estimators):
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, max_depth=max_depth)
    clf.fit(dataset.get(0), dataset.get(1))
    return clf


def make_sklearn_simple_tree(train, labels, tree_out_name):
    # Definition of the classifier
    clf = DecisionTreeClassifier(
        random_state=26525,
        criterion='gini',
        #max_depth=15
        #ccp_alpha=0.015
    )

    clf.fit(train, labels)

    with open("%s.tree" % tree_out_name, "w") as arquivo:
        arquivo.write(tree.export_text(clf, max_depth=1000))


def sklearntree_to_termos(tree_path, qt_inputs):
    ins = qt_inputs

    linha_pla = list()
    control = list()
    pre_termos = list()

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
                # out_class = linha.split(":")[0].replace("|", "")
                out_class = linha.split(":")[1]
                # Por causa das RandomForests!!
                out_class = out_class.replace(".0", "")
                pre_termos.append("%s %s" % ("".join(linha_pla), cifar10_class_to_one_hot(int(out_class))))
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
                            linha_pla[control[-1]] = "-"
                            control.pop(-1)
                        counter = pipes

                linha_pla[attr] = value

    termos = []
    for t in pre_termos:
        termos.append(Termo(t))
    return termos


def sklearntree_to_pla(tree_path, qt_inputs, output_file):
    termos = sklearntree_to_termos("%s" % tree_path, qt_inputs)
    pla_obj = pla_obj_factory(termos, output_file)
    pla_obj.pla_to_file()