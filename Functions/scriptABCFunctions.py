import subprocess
import shlex
import time
import os


def gera_abc_aig_old(arquivo_pla, nome_aig):
    #
    # Comandos no ABC
    # read_pla -->
    # strash   -->
    # balance  -->
    # write    -->
    abc_cmd = subprocess.Popen(shlex.split("./tools/abc -c \'read_pla %s;"
                                                              'strash;'
                                                              'balance;'                                              
                                                              'balance;'
                                                              'balance;'
                                                              'balance;'
                                                              'balance;'                                              
                                                              'balance;'
                                                              'balance;'
                                                              'balance;'
                                                              'balance;'                                              
                                                              'balance;'
                                                              'balance;'
                                                              'balance;'
                                                              "write %s\'" % (arquivo_pla,nome_aig)),
                                           stdout=subprocess.PIPE)
    # O retorno desta funcao sera [0] = stdout e [1] = sterr (erro)
    return abc_cmd.communicate()


def gera_abc_aig(arquivo_pla, nome_aig, cmds=False):
    tempo_inicial = time.time()

    if not cmds:
        comando = "./tools/abc -c \'read_pla %s; strash; write %s\'" % (arquivo_pla, nome_aig)
    else:
        comando = "./tools/abc -c \'read_pla %s; strash; %swrite %s\'" % (arquivo_pla, "".join(["%s; " % i for i in cmds]), nome_aig)

    abc_cmd = subprocess.Popen(shlex.split(comando), stdout=subprocess.PIPE)
    # O retorno desta funcao sera [0] = stdout e [1] = sterr (erro)
    exec_abc = abc_cmd.communicate()
    tempo_final = time.time() - tempo_inicial

    retorno = {"stdout": exec_abc[0],
               "erro": exec_abc[1],
               "tempo": tempo_final}
    return retorno


def faz_resyn2(aig_path, aig_out_path="", substitui_original=False):

    temp_aig = aig_path.replace(".aig", "--resyn2-%s.aig" % str(time.time()))
    original = aig_path

    if not aig_out_path:
        out = temp_aig
    else:
        out = aig_out_path

    comando = "./tools/abc -c \'read %s; resyn2; resyn2; resyn2; resyn2; resyn2; resyn2; resyn2; resyn2; resyn2; " \
              "resyn2; write %s\'" % (aig_path, out)

    abc_cmd = subprocess.Popen(shlex.split(comando), stdout=subprocess.PIPE)
    # O retorno desta funcao sera [0] = stdout e [1] = sterr (erro)
    exec_abc = abc_cmd.communicate()

    if substitui_original:
        os.remove(aig_path)
        os.rename(temp_aig, aig_path)


def pega_valor_ps(saida_abc, string_busca, espaco_campo):
    posicao_inicial = saida_abc.find(string_busca)
    posicao_final = posicao_inicial + len(string_busca)
    valor_campo = int(saida_abc[posicao_final:posicao_final + espaco_campo])
    return valor_campo


def get_abc_ps_aig(aig_path):
    abc_cmd_aig = subprocess.Popen(shlex.split('./tools/abc -c \'&read %s;'
                                               '&ps;\'' % aig_path),
                                   stdout=subprocess.PIPE)
    saida_abc = abc_cmd_aig.communicate()

    # Pegar numero de INPUTS no AIG
    in_outs = b"i/o ="
    posicao_inicial = saida_abc[0].find(in_outs)
    posicao_final = posicao_inicial + len(in_outs)
    numero_ins = int(saida_abc[0][posicao_final:posicao_final + 7])
    numero_outs = int(saida_abc[0][posicao_final + 9:posicao_final + 16])

    # Numero de ANDs
    numero_ands = int(pega_valor_ps(saida_abc[0], b"and =", 8))

    # Pegar a profundidade do AIG
    numero_levs = int(pega_valor_ps(saida_abc[0], b"lev =", 5))

    output = {
        "inputs": numero_ins,
        "outputs": numero_outs,
        "ands": numero_ands,
        "levels": numero_levs,
    }

    return output


def get_mltest_values(aig_path, pla_path):
    valor = abc_mltest_aig(aig_path, pla_path)
    return valor


def abc_mltest_aig(arquivo_aig, arquivo_pla):

    abc_cmd_aig = subprocess.Popen(shlex.split('./tools/abc -c \'&read %s;'
                                                          '&ps;'
                                                          '&mltest %s\'' % (arquivo_aig, arquivo_pla)),
              stdout=subprocess.PIPE)

    # executa o comando e guarda na variavel
    saida_abc = abc_cmd_aig.communicate()

    # Pegar numero de INPUTS no AIG
    in_outs = b"i/o ="
    posicao_inicial = saida_abc[0].find(in_outs)
    posicao_final = posicao_inicial + len(in_outs)
    numero_ins = int(saida_abc[0][posicao_final:posicao_final+7])
    numero_outs = int(saida_abc[0][posicao_final+9:posicao_final+16])

    # Pegar numero de ANDs no AIG
    numero_ands = int(pega_valor_ps(saida_abc[0], b"and =", 8))
    # Pegar a profundidade do AIG
    numero_levs = int(pega_valor_ps(saida_abc[0], b"lev =", 5))
    # Pegar a quantidade de Erros do AIG
    numero_erros = int(pega_valor_ps(saida_abc[0], b"Errors =", 7))
    # Pegar a quantidade de Acertos do AIG
    numero_acertos = int(pega_valor_ps(saida_abc[0], b"Correct =", 7))
    # Pegar a quantidade de vetores que saem 0 -- NAIVE GUESS -- (&mltest)
    numero_naive = int(pega_valor_ps(saida_abc[0], b"Naive guess =", 7))
    # Calcula o total de vetores
    total = numero_acertos + numero_erros
    # Calcula o percentual de acertos
    percentual_acerto = numero_acertos * 100 / total

    output = {
        "inputs": numero_ins,
        "outputs": numero_outs,
        "ands": numero_ands,
        "levels": numero_levs,
        "erros": numero_erros,
        "acertos": numero_acertos,
        "naive": numero_naive,
        "per_acerto": percentual_acerto
    }
    return output


def circuit_abc_map(lib, circ_description, out_name):
    abc_cmd = subprocess.Popen(shlex.split('./tools/abc -c \'read %s;'
                                                            'read %s;'
                                                            'map;'
                                                            'write_verilog %s\'' % (lib, circ_description, out_name)),
                               stdout=subprocess.PIPE)

    exec_abc = abc_cmd.communicate()

    abc_out = {"stdout": exec_abc[0],
               "error": exec_abc[1]}

    return abc_out
