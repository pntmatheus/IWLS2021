import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus


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

if __name__ == "__main__":
    print("Catita")