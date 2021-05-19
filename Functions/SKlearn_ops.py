from Functions.termo import Termo
from Functions.pla import pla_obj_factory

def cifar10_class_to_one_hot(int_class):
    classes = {0: "0000000001",  # airplane
               1: "0000000010",  # automobile
               2: "0000000100",  # bird
               3: "0000001000",  # cat
               4: "0000010000",  # deer
               5: "0000100000",  # dog
               6: "0001000000",  # frog
               7: "0010000000",  # horse
               8: "0100000000",  # ship
               9: "1000000000"}  # truck
    return classes.get(int_class)


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
    return None