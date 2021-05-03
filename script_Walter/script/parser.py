#!/usr/bin/python3

import sys
import os
import re

def main():
  if len(sys.argv) != 2:
    print("Usage: {} abc_log_file.log".format(__file__))
    sys.exit()

  filepath = sys.argv[1]
  
  print(filepath + ',', end=' ')

  if not os.path.isfile(filepath):
    print("File path {} does not exist. Exiting...".format(filepath))
    sys.exit()

  with open(filepath) as fp:
    for line in fp:
      #ex00.train : i/o =     32/      1  and =     601  lev =   15 (15.00)  mem = 0.01 MB
      headerLine = re.match("^.*i/o =\s+(\d+)/\s+(\d+).*and =\s+(\d+).*lev =\s+(\d+).*$", line)
      if headerLine:
        inputs  = headerLine.group(1)
        outputs = headerLine.group(2)
        ands    = headerLine.group(3)
        levels  = headerLine.group(4)
        print(inputs + ", " + outputs + ", " + ands + ", " + levels + ', ', end=' ')

      #Finished reading 6400 simulation patterns for 32 inputs. Probability of 1 at the output is  49.41 %.
      finishedLine = re.match("^Finished reading (\d+) simulation patterns for (\d+) inputs\. Probability of 1 at the output is\s+(\d+\.\d+).*$", line)
      if finishedLine:
        patterns    = finishedLine.group(1)
        inputs      = finishedLine.group(2)
        probability = finishedLine.group(3)
        print(patterns + ", ", end=' ')

      #Total =   6400.  Errors =    758.  Correct =   5642.  ( 88.16 %)   Naive guess =   3238.  ( 50.59 %)
      totalLine = re.match("^Total =\s+(\d+).*Errors =\s+(\d+).*Correct =\s+(\d+).*?(\d+\.\d+).*Naive guess =\s+(\d+).*?(\d+\.\d+).*$", line)
      if totalLine:
        total                = totalLine.group(1)
        errors               = totalLine.group(2)
        correct              = totalLine.group(3)
        correctPercentage    = totalLine.group(4)
        naiveGuess           = totalLine.group(5)
        naiveGuessPercentage = totalLine.group(6)
        print(errors+ ", " + correct+ ", " + correctPercentage+ ", " + naiveGuess+ ", " + naiveGuessPercentage)
  print()

if __name__ == '__main__':
  main()
