import os
import subprocess
import numpy as np
import pandas as pd
import pydotplus
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, make_scorer
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split

def read_iop(fn):
  # Reads .i, .o, and .p parameters from the pla benchmark.
  i = o = p = 0
  f = open(fn, 'r')
  for l in f:
    if l.startswith('.i'):
      i = int(l[3:])
    elif l.startswith('.o'):
      o = int(l[3:])
    elif l.startswith('.p'):
      p = int(l[3:])
    elif not l.startswith('.'):
      break
  return i, o, p

def tt2df(fn):
  # Loads a pla bechmark into a Pandas dataframe.
  # The parameter fn is the path of the pla file.
  tt = []
  content = np.genfromtxt(fn, skip_header=4, skip_footer=1, dtype='str')
  for idx in range(0, content.shape[0]):
    tt.append(list(content[idx, 0] + content[idx, 1]))
  return pd.DataFrame(tt)

def draw_dtc(dtc, x, y):
  # This function can be used to visualize the classification tree (if needed).
  # However, very large classification tree take a huge amount or resources.
  dot_data = StringIO()
  export_graphviz(
      dtc,
      out_file=dot_data,
      filled=True,
      rounded=True,
      special_characters=True,
      feature_names=x,
      class_names=y)
  graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
  return (Image(graph.create_png()))


to_improve = ["ex48.train.pla", "ex49.train.pla", "ex42.train.pla", "ex26.train.pla", "ex74.train.pla", "ex47.train.pla", "ex06.train.pla", "ex08.train.pla", "ex02.train.pla", "ex44.train.pla", "ex04.train.pla", "ex28.train.pla", "ex46.train.pla", "ex45.train.pla", "ex29.train.pla", "ex22.train.pla", "ex24.train.pla", "ex27.train.pla", "ex20.train.pla", "ex25.train.pla", "ex90.train.pla", "ex94.train.pla", "ex78.train.pla", "ex91.train.pla", "ex00.train.pla", "ex96.train.pla", "ex98.train.pla", "ex97.train.pla", "ex92.train.pla", "ex99.train.pla", "ex52.train.pla"]

# Define the target benchmark
for i in range(100):
    benchmark_id = str(i).zfill(2)
    workDir = os.getcwd()
    
    validation_name = 'ex{}.valid.pla'.format(benchmark_id)
    trainning_name  = 'ex{}.train.pla'.format(benchmark_id)
    verilog = open(workDir + '/ex{}.v'.format(benchmark_id), 'w') 
    aig = workDir + '/ex{}.aig'.format(benchmark_id)
    print('Output file format: ' + workDir + '/ex{}.v')
    print('AIG file', aig)
    # In case file accuracy is < 70% (pre-computed)
    if(trainning_name in to_improve):
        print('Handling ' + trainning_name + ' as special case!')
        
        merged = open(workDir + '/ex{}_merged.train.pla'.format(benchmark_id), 'w') 
        with open(workDir + '/ex{}.train.pla'.format(benchmark_id)) as train:
            for line in train:
                if(line.startswith('.e')):
                      continue
                if(line.startswith('.p')):
                    line = line.replace('.p 6400', '.p 12800')
                    merged.write(line)
                    continue
                    
                merged.write(line)
        train.close()
        
        with open(workDir + '/ex{}.valid.pla'.format(benchmark_id)) as valid:
            for line in valid:
                if(line.startswith('.')):
                      continue
                merged.write(line)
            merged.write('.e')
        valid.close()
        merged.close()
        
        trainning_name = 'ex{}_merged.train.pla'.format(benchmark_id)

    ############################################################################################
    # Read training and validation benchmarks
    i, o, p = read_iop(workDir + '/' + trainning_name)
    ts = tt2df(workDir + '/' + trainning_name)
    vs = tt2df(workDir + '/' + validation_name)

    print('Trainning file: ex{}.train.pla'.format(benchmark_id))
    
    # Split features and the target variable
    X_train = ts.iloc[:, :-1].values
    y_train = ts.iloc[:, -1].values
    X_val = vs.iloc[:, :-1].values
    y_val = vs.iloc[:, -1].values

    # Understand if classes are unbalanced (it might give some insights on how the clf performs)
    print('1\'s:', np.count_nonzero(y_train == '1'))
    print('0\'s:', np.count_nonzero(y_train == '0'))

    # Definition of the classifier
    clf = DecisionTreeClassifier(
      random_state=9856230,
      criterion='gini',
      max_depth=8,
    )

    # Training and validation of the classifier
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_val)
    y_predicted_training = clf.predict(X_train)

    print(trainning_name + ', ' + '{:.4f}'.format(accuracy_score(y_train, y_predicted_training)) + ', ' + '{:.4f}'.format(accuracy_score(y_val, y_predicted)))

    # Generate a Verilog description of the classifier.
    # TODO: create a function out of this code snippet.
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    verilog.write('module top (')
    verilog.write('\t' + ', '.join(['x{}'.format(v) for v in np.arange(0, i)]) + ', ')
    verilog.write('\t' + 'y);\n')
    verilog.write('input ' + ', '.join(['x{}'.format(v) for v in np.arange(0, i)]) + ';\n')
    verilog.write('output y;\n')
    verilog.write('wire ' + ', '.join(['n{}'.format(v) for v in np.arange(0, n_nodes)]) + ';\n')
    for i in range(n_nodes):
        if is_leaves[i]:
            verilog.write('assign n{node} = {out_class};\n'.format(
                node=i,
                out_class=np.argmax(clf.tree_.value[i])
            ))
        else:
            verilog.write('assign n{node} = x{feature} ^ 1\'b1 ? n{left} : n{right};\n'.format(
                node=i,
                feature=feature[i],
                left=children_left[i],
                right=children_right[i]
            ))
    verilog.write('assign y = n0;\n')
    verilog.write('endmodule')
    verilog.close()
    #Uncomment the following line if the abc exec is in the current directory
    #cmd = './abc -c "r ' + workDir + '/ex{}.v'.format(benchmark_id) + '; st; resyn2; compress2; resyn3; compress; rwsat; ps; write_aiger ' + aig + '"'
    cmd = 'abc -c "r ' + workDir + '/ex{}.v'.format(benchmark_id) + '; st; resyn2; compress2; resyn3; compress; rwsat; ps; write_aiger ' + aig + '"'
    os.system(cmd)
    print(cmd)
