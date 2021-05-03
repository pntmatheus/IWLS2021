#!/bin/bash

# creates classification tree and outputs .aig and .v 
python3 ct-contest.py

# validates each .aig with ABC and outputs a .log file for each 
CURRENT=$(pwd)
ABC=abc
for file in *.aig; do
  BASE="${file##*/}"
  BASE="${BASE%.aig}"
  echo "Validating file ${file} with ${BASE}.valid.pla"
  
  ABC_CMD="&read ${file}; &ps; &mltest ${BASE}.valid.pla"
  ABC_OUTPUT=${BASE}_ABC.log
  #Uncomment the following line if the abc exec is in the current directory
  #"${CURRENT}/ABC" -c "$ABC_CMD" > $ABC_OUTPUT
  ABC -c "$ABC_CMD" > $ABC_OUTPUT
  tail -n 1 ${ABC_OUTPUT}
done

echo "ins, outs, ands, levels, #patterns, errors, correct, correct %, naive, naive %" > final_metrics.csv
for file in *.log; do
  python3 parser.py $file >> final_metrics.csv
done