import pexpect
import argparse

import os
import shutil

EQN_HEADER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'EQN_header_abc_nomenc.eqn'
)

def replace_last(
    source_string: str,
    replace_what: str,
    replace_with: str
) -> str:
    head, _sep, tail = source_string.rpartition(replace_what)
    return head + replace_with + tail

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Junta todos os estimators com o votador')

    parser.add_argument(
        'aigsFolder',
        metavar='aigs_folder',
        type=str,
        help='pasta com os AIGs dos estimators individuais'
    )

    parser.add_argument(
        'eqnVoter',
        metavar='eqn_voter',
        type=str,
        help='arquivo com o EQN do votador'
    )

    parser.add_argument(
        'outFile',
        metavar='out_file',
        type=str,
        help='nome do AIG de saída'
    )

    parser.add_argument(
        '-d',
        type = int,
        default = 2733,
        help = 'número de linhas para remover do cabeçalho dos EQNs'
    )

    args = parser.parse_args()

    outFile = args.outFile
    if (not args.eqnVoter.endswith('.aig')):
        outFile = outFile + '.aig'
    
    aigsFiles = [x for x in os.scandir(args.aigsFolder)]
    aigsFiles = sorted(list(filter(
        lambda x: x.endswith('.aig'), [x.name for x in aigsFiles]
    )))

    if len(aigsFiles) == 0:
        print('nenhum AIG encontrado')
        exit(-1)

    try:
        os.mkdir(os.path.join(args.aigsFolder, '.eqn'))
    except FileExistsError:
        pass

    print('convertendo AIGs para EQNs...    ', end='')


    # aigsFiles = aigsFiles[0:1]

    for aig in aigsFiles:
        #cmd = "run/abc/abc -c 'read {0}; write {1};'".format(
        cmd = "../tools/abc -c 'read {0}; write {1};'".format(
            os.path.join(args.aigsFolder, aig),
            os.path.join(args.aigsFolder, '.eqn', replace_last(aig, '.aig', '.eqn'))
        )

        proc = pexpect.spawn(cmd, encoding='utf-8')
        responseLines = str(proc.read())
        proc.wait()
        if (proc.exitstatus != 0):
            print('')
            print(responseLines)
            exit(-1)
    print('ok')

    print('gerando EQN único e adicionando o votador...    ', end='')
    eqnFiles = [
        os.path.join(args.aigsFolder, '.eqn', replace_last(x, '.aig', '.eqn')) for
            x in aigsFiles
    ]
    
    for i, eqn in enumerate(eqnFiles):
        replaceOutputs = ';'.join([
            's/po{0}/C{0}t{1}/g'.format(x, i) for x in range(10)
        ])
        cmd = "sed -i '1,{0}d;$d;s/new_n/t{1}n/g;{3}' {2}"\
            .format(args.d, i, eqn, replaceOutputs)

        proc = pexpect.spawn(cmd, encoding='utf-8')
        responseLines = str(proc.read())
        proc.wait()
        if (proc.exitstatus != 0):
            print('')
            print(responseLines)
            exit(-1)

    cmd = "/bin/bash -c 'cat {0} {1} {2} > {3}'".format(
        EQN_HEADER,
        args.eqnVoter,
        ' '.join(eqnFiles),
        os.path.join(args.aigsFolder, '.eqn', '.out.eqn')
    )
    proc = pexpect.spawn(cmd, encoding='utf-8')
    responseLines = str(proc.read())
    proc.wait()
    if (proc.exitstatus != 0):
        print('')
        print(responseLines)
        exit(-1)
    print('ok')
    
    print('convertendo EQN único para o AIG final...    ', end='')
    #cmd = "run/abc/abc -c 'read {0}; strash; write {1};'".format(
    cmd = "../tools/abc -c 'read {0}; strash; write {1};'".format(
        os.path.join(args.aigsFolder, '.eqn', '.out.eqn'),
        outFile
    )
    proc = pexpect.spawn(cmd, encoding='utf-8')
    responseLines = str(proc.read())
    proc.wait()
    if (proc.exitstatus != 0):
        print('')
        print(responseLines)
        exit(-1)
    print('ok')

    
    print('apagando arquivos intermediários...    ', end='')
    shutil.rmtree(os.path.join(args.aigsFolder, '.eqn'))
    print('ok')

