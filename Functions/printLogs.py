import os

if __name__ == "__main__":

    for root, dirs, files in os.walk("../Contest_AIGs_INFO", topdown=False):
        files.sort()
        for name in files:
            with open(os.path.join(root, name), "r") as arquivo:
                linhas = arquivo.readlines()
                for linha in linhas:
                    if "OK!!" not in linha:
                        print("%s" % (linha.replace("\n", "")))
