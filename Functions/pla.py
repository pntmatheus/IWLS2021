from __future__ import annotations

import re
import os
from typing import Optional, List
from Functions.termo import Termo
from multipledispatch import dispatch


@dispatch(str)
def pla_obj_factory(pla_path):
    print("papai")
    """ Função que cria e retorna um objeto PLA

    Args:
        pla_path(str): caminho onde está o arquivo PLA

    Return:
        Um objeto do tipo PLA
     """
    nome = str(pla_path).split(os.path.sep)[-1]
    nome = nome.replace(".pla", "")

    with open(pla_path, "r") as arquivo:
        pla_path = arquivo.read()

    # Regex patterns
    in_pat = re.compile('\.i(.*?)\\n')
    out_pat = re.compile('\.o(.*?)\\n')
    type_pat = re.compile('\.type(.*?)\\n')
    termos_pat = re.compile('\.p(.*?)\\n')
    coment_pat = re.compile('#(.*?)\\n')
    pulou_linha_pat = re.compile('(.*?)\\n')
    so_termos = re.compile('[^-\d\n\s].*')

    # Remover comentarios (se existirem!!)
    pla_sem_comentario = re.sub(coment_pat, "", pla_path)

    qt_ins = re.search(in_pat, pla_sem_comentario).group(1)
    qt_outs = re.search(out_pat, pla_sem_comentario).group(1)
    qt_termos = re.search(termos_pat, pla_sem_comentario).group(1)
    termos = re.findall(pulou_linha_pat, pla_sem_comentario)

    pla_type = re.search(type_pat, pla_sem_comentario)
    if pla_type is not None:
        pla_type = pla_type.group(1)
        pla_type = pla_type.replace(" ", "")

    qt_ins = int(qt_ins)
    qt_outs = int(qt_outs)
    qt_termos = int(qt_termos)

    lista_exclusao = list()

    for termo in termos:
        if bool(re.match(so_termos, termo)):
            lista_exclusao.append(termo)

    for termo in lista_exclusao:
        termos.remove(termo)

    termos_obj = list()

    for termo in termos:
        termos_obj.append(Termo(termo))

    pla_dict = {"qt_ins": qt_ins,
                "qt_outs": qt_outs,
                "type": pla_type,
                "qt_termos": qt_termos,
                "termos": termos}

    pla_obj = Pla(nome, pla_dict["type"], pla_dict["qt_ins"], pla_dict["qt_outs"], termos_obj)
    return pla_obj


@dispatch(list, str)
def pla_obj_factory(termos: List[Termo], name) -> Pla:
    if termos is not None:
        qt_in = termos[0].get_qt_input()
        qt_out = termos[0].get_qt_output()
        pla_obj = Pla(name, None, qt_in, qt_out, termos)
    else:
        pla_obj = None

    return pla_obj


class Pla:

    def __init__(self, name, tipo, qt_inputs, qt_outputs, termos):
        self.nome = name
        self.type = tipo
        self.qt_inputs = qt_inputs
        self.qt_outputs = qt_outputs
        self.termos = termos  # type: List[Optional[Termo]]

    def __str__(self):
        return self.nome

    def get_nome(self):
        return self.nome

    def get_qt_inputs(self):
        return self.qt_inputs

    def get_qt_outputs(self):
        return self.qt_outputs

    def get_type(self):
        return self.type

    def get_total_pla_0(self):
        return self.__get_total__("0")

    def get_total_pla_1(self):
        return self.__get_total__("1")

    def get_total_pla_dcare(self):
        return self.__get_total__("-")

    def __get_total__(self, variavel):
        total = 0
        for termo in self.termos:
            total = total + termo.get_input().count(variavel)
        return total

    def get_termos(self):
        return self.termos

    def get_termos_unique(self):
        unique = set()
        for t in self.termos:
            unique.add(t)
        return unique

    def turn_termos_unique(self):
        """ Remove os termos do pla que são iguais e atualiza o atributo \'termos\'"""
        self.termos = list(self.get_termos_unique())

    def get_total_termos(self):
        return len(self.termos)

    def remove_repetidos(self):
        self.termos = list(dict.fromkeys(self.termos))

    def get_offsets(self):
        return self.get_termos_by_output("0")

    def get_onsets(self):
        return self.get_termos_by_output("1")

    def get_termos_by_output(self, output):
        termos_output = list()
        for termo in self.termos:
            if termo.output == output:
                termos_output.append(termo)
        return termos_output

    def pla_to_file(self, output_path=None, out_name=None, in_labels=None, out_labels=None):
        output_file = ""
        path = ""

        if out_name is not None:
            output_file = out_name
        else:
            output_file = self.nome

        if output_path is not None:
            path = "%s/%s.pla" % (output_path, output_file)
        else:
            path = "%s.pla" % output_file

        with open(path, "w") as pla_file:
            pla_file.write(".i %d\n" % self.get_qt_inputs())
            pla_file.write(".o %d\n" % self.get_qt_outputs())

            # pla_lines.append("%s %s" % ("".join(["%s" % i for i in inputs])
            if in_labels is not None:
                labels = "".join(["%s " % i for i in in_labels])
                labels = labels[:-1]
                pla_file.write(".ilb %s\n" % labels)
            if out_labels is not None:
                labels = "".join(["%s " % i for i in out_labels])
                labels = labels[:-1]
                pla_file.write(".ob %s\n" % labels)
            pla_file.write(".p %d\n" % len(self.termos))

            if self.type is None:
                pla_file.write(".type fr\n")
            else:
                pla_file.write(".type %s" % self.type)

            for t in self.termos:
                pla_file.write("%s %s\n" % (t.get_input(), t.get_output()))

            pla_file.write(".e\n")