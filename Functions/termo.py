class Termo:

    def __init__(self, linha_pla):
        self.termo = linha_pla
        self.input = linha_pla.split(" ")[0]
        self.output = linha_pla.split(" ")[1]

    def __str__(self):
        return self.termo

    def __eq__(self, other):
        return (self.input, self.output) == (other.input, other.output)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.input,self.output))

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output

    def get_total_0(self):
        total = self.__conta_variavel__("0")
        return total

    def get_total_1(self):
        total = self.__conta_variavel__("1")
        return total

    def get_total_dcare(self):
        total = self.__conta_variavel__("-")
        return total

    def __conta_variavel__(self, variavel):
        return self.input.count(variavel)
