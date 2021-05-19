import re
import subprocess
import time
import os
import shutil

from Functions.scriptABCFunctions import get_mltest_values, get_abc_ps_aig, faz_resyn2
from Functions.scriptABCFunctions import gera_abc_aig, abc_mltest_aig
from Functions.pla import pla_obj_factory
from subprocess import check_output


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
    for i in range(pla.get_qt_outputs()):
        conteudo.append("@attribute output {0,1}")

    # Adicionar o dataset
    conteudo.append("@data")
    for t in pla.get_termos():
        termo_format = "%s%s" % (
            "".join(["%s," % i for i in t.get_input()]), "".join(["%s," % i for i in t.get_output()]))
        conteudo.append(termo_format[:-1])

    if gera_arquivo:
        with open("%s/%s.arff" % (path,pla.get_nome()), "w") as arquivo:
            for i in conteudo:
                arquivo.write(i + "\n")


def wekatree_to_termos(wekatree, qt_inputs):
    ins = qt_inputs

    linha_pla = list()
    control = list()
    termos = list()

    for j in range(ins):
        linha_pla.append("-")

    counter = -1
    for linha in wekatree:
        attr, value = linha.replace(" ", "").split("=")
        pipes = attr.count("|")
        # Pega apenas o número da feature
        attr_num = int(re.sub("\D", "", attr))

        if counter < pipes:
            control.append(attr_num)
            counter = counter + 1
        else:
            if counter > pipes:
                for i in range(counter - pipes):
                    linha_pla[control[-1]] = "-"
                    control.pop(-1)
                counter = pipes

        if ":" in value:
            saida = value.split(":")
            linha_pla[attr_num] = saida[0]
            termos.append("%s %s" % ("".join(linha_pla), saida[1].split("(")[0]))
        else:
            linha_pla[attr_num] = value

    return termos


def get_weka_output(pla_obj, options, classificador="J48", faz_cross=False):
    weka_default_call = ['java', '-jar', 'tools/weka_java.jar', '../tmp_iwls2020/ARFF/%s.arff' % pla_obj.get_nome(),
                         classificador]
    if faz_cross:
        cross = "true"
    else:
        cross = "false"

    weka_default_call.append(cross)

    for opt in options:
        weka_default_call.append(opt)
    print(weka_default_call)
    weka_output = check_output(weka_default_call, stderr=subprocess.DEVNULL)

    return weka_output


def get_weka_output_wrapper(classifier, pla_obj, option, fazCross=False, gera_arquivo=False, nome_arquivo=""):
    if classifier == "PART":
        return get_weka_part_output(pla_obj, option, fazCross, gera_arquivo, nome_arquivo)
    else:
        return get_weka_j48_output(pla_obj, option, fazCross, gera_arquivo, nome_arquivo)


def get_weka_part_output(pla_obj, options, faz_cross=False, gera_arquivo=False, nome_arquivo=""):

    part_output = get_weka_output(pla_obj, options, "PART", faz_cross)

    if gera_arquivo:
        if not nome_arquivo:
            nome_arquivo = "%s-%s" % (pla_obj.get_nome(), "".join(options))

        with open("tmp_iwls2020/PART/%s.part" % nome_arquivo, "w") as arquivo:
            arquivo.write(part_output.decode())

    return {"output": part_output, "nome_arquivo": nome_arquivo}


def get_weka_j48_output(pla_obj, options, faz_cross=False, gera_arquivo=False, nome_arquivo=""):

    j48_output = get_weka_output(pla_obj, options, "J48", faz_cross)

    if gera_arquivo:
        if not nome_arquivo:
            nome_arquivo = "%s-%s" % (pla_obj.get_nome(), "".join(options))

        with open("tmp_iwls2020/J48/%s.j48" % nome_arquivo, "w") as arquivo:
            arquivo.write(j48_output.decode())

    return {"output": j48_output, "nome_arquivo": nome_arquivo}


def to_aig_wrapper(classifier, pla_obj, nome_arquivo):

    if classifier == "PART":
        part_to_aig(pla_obj, nome_arquivo)
        return "tmp_iwls2020/PART_AIG/%s.aig" % nome_arquivo
    else:
        j48pla = j48tree_to_pla("%s.j48" % nome_arquivo,
                                pla_obj.get_qt_inputs(),
                                pla_obj.get_qt_outputs())[0]
        j48pla_to_aig("tmp_iwls2020/J48_PLA/%s" % j48pla, "tmp_iwls2020/J48_AIG/", True)
        return "tmp_iwls2020/J48_AIG/%s.aig" % nome_arquivo


def part_to_aig(pla_obj, nome_part):
    comando = ['./tools/parttoaag', str(pla_obj.get_qt_inputs()), nome_part]
    output = check_output(comando, stderr=subprocess.DEVNULL)

    comando2 = ['./tools/aigtoaig', "tmp_iwls2020/AAG/%s.aag" % nome_part, "tmp_iwls2020/PART_AIG/%s.aig" % nome_part]
    output2 = check_output(comando2, stderr=subprocess.DEVNULL)


def j48pla_to_aig(j48pla_path, path="../tmp_iwls2020/J48_AIG/", faz_resyn2=False):
    if faz_resyn2:
        resyns = ["resyn2", "resyn2", "resyn2", "resyn2", "resyn2", "resyn2", "resyn2", "resyn2", "resyn2", "resyn2"]
        return gera_abc_aig(j48pla_path, "%s%s" % (path, j48pla_path.split("/")[-1].replace(".pla", ".aig")), resyns)
    else:
        return gera_abc_aig(j48pla_path, "%s%s" % (path, j48pla_path.split("/")[-1].replace(".pla", ".aig")))


def j48tree_to_pla(j48_path, pla_ins, pla_outs, pla_path="tmp_iwls2020/J48_PLA/", path="tmp_iwls2020/J48/"):
    with open("%s%s" % (path, j48_path), "r") as arquivo:
        weka_output = arquivo.read().encode().split(b"\n\n")
        if b"\n------------------\n:" in weka_output[0]:
            conteudo_pre = list()
            conteudo_pre.append("")
            for i in range(pla_ins):
                conteudo_pre[0] = conteudo_pre[0] + "-"
            conteudo_pre[0] = "%s 1" % conteudo_pre[0]
            leaves = 1
            size_tree = 1
        else:
            conteudo_pre = wekatree_to_termos(weka_output[1].decode().splitlines(), pla_ins)
            leaves = int(re.sub("\D", "", weka_output[2].decode()))
            size_tree = int(re.sub("\D", "", weka_output[3].decode()))

        weka_pla_nome = "%s" % (arquivo.name.split("/")[-1].replace(".j48", ".pla"))

        with open("%s" % pla_path + weka_pla_nome, "w") as pla:
            pla.write(".i %d\n" % pla_ins)
            pla.write(".o %d\n" % pla_outs)
            pla.write(".p %d\n" % len(conteudo_pre))
            pla.write(".type fr\n")
            for i in conteudo_pre:
                pla.write(i + "\n")
            pla.write(".e")

        # Nome do arquivo gerado;
        # Numero de folhas da arvore
        # Tamanho da arvore
        return [weka_pla_nome,
                leaves,
                size_tree]


def j48_to_aig(pla_obj, options, path="tmp_iwls2020/J48_PLA/"):
    j48 = j48_to_pla(pla_obj, options, path)
    nome_pla = j48[0]
    gera_abc_aig("%s%s.pla" % (path, nome_pla), "tmp_iwls2020/J48_AIG/%s.aig" % nome_pla)
    return "%s%s.aig" % (path, nome_pla)


def j48_to_pla(pla_obj, options, path="tmp_iwls2020/NEURAL_PLAs/"):

    #weka_output = get_weka_output(pla_obj, options, "J48", False)

    weka_output = get_weka_j48_output(pla_obj, options, False, False, "")["output"]

    weka_output = weka_output.split(b"\n\n")

    if b"\n------------------\n:" in weka_output[0]:
        conteudo_pre = list()
        conteudo_pre.append("")
        for i in range(pla_obj.get_qt_inputs()):
            conteudo_pre[0] = conteudo_pre[0] + "-"
        conteudo_pre[0] = "%s 1" % conteudo_pre[0]
        leaves = 1
        size_tree = 1
    else:
        conteudo_pre = wekatree_to_termos(weka_output[1].decode().splitlines(), pla_obj.get_qt_inputs())
        leaves = int(re.sub("\D", "", weka_output[2].decode()))
        size_tree = int(re.sub("\D", "", weka_output[3].decode()))

    weka_pla_nome = "%s-J48%s" % (pla_obj.get_nome(), "".join(options))

    with open("%s" % path + weka_pla_nome + ".pla", "w") as pla:
        pla.write(".i %d\n" % pla_obj.get_qt_inputs())
        pla.write(".o %d\n" % pla_obj.get_qt_outputs())
        pla.write(".p %d\n" % len(conteudo_pre))
        pla.write(".type fr\n")
        for i in conteudo_pre:
            pla.write(i + "\n")
        pla.write(".e")

    # Nome do arquivo gerado;
    # Numero de folhas da arvore
    # Tamanho da arvore
    return [weka_pla_nome,
            leaves,
            size_tree]


def get_cross_valid_info(weka_out):
    retorno = weka_out.decode()
    retorno = retorno.split("###$###")[1]
    retorno = retorno.replace("\n", "")
    retorno = retorno.split(";;")
    return retorno


def verifica_cross_validation(weka_output):
    print(weka_output)


def compara_weka_to_aig(pla_obj, weka_options):
    weka_pla = j48_to_pla(pla_obj, weka_options)

    neural_pla, num_leaves, tree_size = weka_pla

    neural_pla_obj = pla_obj_factory("tmp_iwls2020/NEURAL_PLAs/%s.pla" % neural_pla)

    gera_abc_aig("tmp_iwls2020/NEURAL_PLAs/%s.pla" % neural_pla_obj.nome,
                 "tmp_iwls2020/NEURAL_AIGs/%s.aig" % neural_pla_obj.nome)

    get_pla_ex = pla_obj.get_nome().split(".")[0]
    mltest_table = abc_mltest_aig("tmp_iwls2020/NEURAL_AIGs/%s.aig" % neural_pla_obj.nome,
                                  "PLA_files/%s.valid.pla" % get_pla_ex)
    mltest_train = abc_mltest_aig("Resultados/WEKA/NEURAL_AIGs/%s.aig" % neural_pla_obj.nome,
                                  "PLA_files/%s.train.pla" % get_pla_ex)

    linha = "%s;;%s;;%s;;%s;;%s;;%s;;%d;;%d;;%d;;%s.pla;;%s.aig;;%s.arff" % (num_leaves,
                                                                            tree_size,
                                                                            mltest_table["ands"],
                                                                            mltest_table["per_acerto"],
                                                                            mltest_train["per_acerto"],
                                                                            "###RUNTIME###",
                                                                            len(neural_pla_obj.get_onsets()),
                                                                            len(neural_pla_obj.get_offsets()),
                                                                            len(neural_pla_obj.get_termos()),
                                                                            neural_pla_obj.get_nome(),
                                                                            neural_pla_obj.get_nome(),
                                                                            pla_obj.get_nome())

    return linha


def gera_tabela_neural_test_options():
    foo_list = list()
    min_objects = ["0", "1", "2", "3", "4", "5", "10"]
    confidence = ["0.0001", "0.001", "0.01", "0.1", "0.25", "0.5"]
    last_options = ["-U", "-M", "0", "-J"]

    for i in range(99, 100):
        for versao in ["train", "valid", "total"]:

            nome_pla = "ex%02d.%s" % (i, versao)
            print("\n\n\n\n")
            print("######### %s ############" % nome_pla)

            pla_obj = pla_obj_factory("PLA_files/%s.pla" % nome_pla)

            pla_obj_to_arff(pla_obj, True, True)

            for m in min_objects:
                for c in confidence:
                    tempo_weka = time.time()
                    linha = compara_weka_to_aig(pla_obj, ["-M", m, "-C", c])
                    linha = linha.replace("###RUNTIME###", "%.2f" % (time.time() - tempo_weka))
                    foo_list.append(linha)
                    print(linha)

            tempo_weka = time.time()
            linha = compara_weka_to_aig(pla_obj, last_options)
            linha = linha.replace("###RUNTIME###", "%.2f" % (time.time() - tempo_weka))
            print(linha)
            foo_list.append(linha)

    with open("Resultados/WEKA/resultadosNeural.txt", "w") as file:
        for f in foo_list:
            file.write(f + "\n")


def verifica_melhor_aig(linhas, cross_index, and_index, path_index):
    flag = 0.0
    melhor_aig = ""
    qt_ands = 0

    for l in linhas:
        dados = l.split(";;")

        # Por causa do JAVA no Medusa da UFPEL...
        dados[cross_index] = dados[cross_index].replace(",", ".")
        cross_v = float(dados[cross_index])
        if cross_v > flag:
            if int(dados[and_index]) < 5000:
                melhor_aig = dados[path_index]
                qt_ands = int(dados[and_index])
                flag = cross_v
        else:
            if cross_v == flag:
                if int(dados[and_index]) > qt_ands:
                    melhor_aig = dados[path_index]
                    qt_ands = int(dados[and_index])
                    flag = cross_v

    return melhor_aig


def gera_linhas_iwls2020_output(options, classifiers, pla_obj, mltest_list):

    def format_mltest_results(aig_path, mltest_list):
        values = list()
        r_value = ""
        for m in mltest_list:
            values.append(get_mltest_values(aig_path, m)["per_acerto"])

        for v in values:
            r_value = "%s%s;;" % (r_value, str(v))

        return r_value

    linhas = list()

    for classifier in classifiers:
        for opt in options:
            tempo = time.time()
            weka_out = get_weka_output_wrapper(classifier,
                                               pla_obj,
                                               opt,
                                               fazCross=True,
                                               gera_arquivo=True,
                                               nome_arquivo="")

            cross_validation = get_cross_valid_info(weka_out["output"])
            aig_path = to_aig_wrapper(classifier, pla_obj, weka_out["nome_arquivo"])
            faz_resyn2(aig_path, substitui_original=True)
            aig_stats = get_abc_ps_aig(aig_path)

            if mltest_list:
                mltests_value = format_mltest_results(aig_path, mltest_list)
                tempo = time.time() - tempo
                linha = "%s;;%s;;%s%s;;%.2f;;%s" % (cross_validation[0],
                                                    aig_stats["ands"],
                                                    mltests_value,
                                                    aig_stats["levels"],
                                                    tempo,
                                                    aig_path)
            else:
                tempo = time.time() - tempo
                linha = "%s;;%s;;%s;;%.2f;;%s" % (cross_validation[0],
                                                  aig_stats["ands"],
                                                  aig_stats["levels"],
                                                  tempo,
                                                  aig_path)

            linhas.append(linha)
    return linhas


def gera_melhor_aig(pla_obj,
                    mltest_list='',
                    out_filename='',
                    verbose=False, supress=False, persist=False, tmp_clean=False):

    tempo_inicial = time.time()

    linhas = list()
    pla_obj_to_arff(pla_obj, True, False)

    options = [["-M", "2", "-C", "0.001"],
               ["-M", "2", "-C", "0.01"],
               ["-M", "2", "-C", "0.1"],
               ["-M", "2", "-C", "0.25"],
               ["-M", "2", "-C", "0.5"]]

    linhas = gera_linhas_iwls2020_output(options, ["J48", "PART"], pla_obj, mltest_list)

    melhor = verifica_melhor_aig(linhas, 0, 1, -1)

    # Pega o melhor C do melhor AIG até aqui!!
    result = re.search('-M2-C(.*)\.aig', melhor)
    best_c = result.group(1)

    # J48 ou PART?
    if "PART_AIG" in melhor:
        best_classifier = "PART"
    else:
        best_classifier = "J48"

    dynamic_opts = [["-M", "0", "-C", best_c],
                    ["-M", "1", "-C", best_c],
                    ["-M", "3", "-C", best_c],
                    ["-M", "4", "-C", best_c],
                    ["-M", "5", "-C", best_c],
                    ["-M", "10", "-C", best_c]]

    new_lines = gera_linhas_iwls2020_output(dynamic_opts, [best_classifier], pla_obj, mltest_list)

    linhas = linhas + new_lines

    melhor = verifica_melhor_aig(linhas, 0, 1, -1)

    for a, linha in enumerate(linhas):
        if melhor in linha:
            linhas[a] = linhas[a] + ";;<====BEST_OPTION"

    if out_filename:
        filename = out_filename
    else:
        filename = "%s.aig" % pla_obj.get_nome()

    shutil.copy(melhor, filename)

    if persist:
        shutil.copytree("tmp_iwls2020", "persist_iwls2020", dirs_exist_ok=True)

    if verbose:
        for linha in linhas:
            print(linha)

    # Clean tmp_iwls2020 directory
    if not tmp_clean:
        for root, dirs, files in os.walk("tmp_iwls2020", topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))

    if not supress:
        print("AIG successfully created: %s" % os.path.realpath(filename))
        print("Runtime (secs) %f " % (time.time() - tempo_inicial))

    print("%s OK!!" % pla_obj.get_nome())
