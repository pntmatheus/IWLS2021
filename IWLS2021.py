from Functions.pla import pla_obj_factory
from Functions.wekaWrapper import j48tree_to_pla


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


if __name__ == "__main__":
    pla_obj = pla_obj_factory("IWLS2021-files/pla_cifar10_chunk_12500.pla")
    j48tree_to_pla("../../tools/pla_memory_chunk_12500.j48", pla_obj.get_qt_inputs(), pla_obj.get_qt_outputs(), "tools/")
    #pla_obj_to_arff(pla_obj, True, True, "IWLS2021-files")

