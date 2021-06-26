Dependências:
pexpect (para executar binários de dentro do script)

Como usar:
$ python3.8 applyVoter.py <pasta com as arvores> <votador_X_arvores>.eqn <nome do aig final>

Importante: Não colocar o aig final na mesma pasta das arvores originais,
porque se for o script for executado de novo ele vai considerar o novo AIG
como uma outra arvore e unir todo mundo.
