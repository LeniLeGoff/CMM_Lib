#!\usr\bin\env python

import sys
import os
import subprocess as sp

if len(sys.argv) != 2:
	print("Usage : folder of archive experiments")
	print("dataset file")
  	sys.exit(1)

for folder in os.listdir(sys.argv[1]) :
	path = sys.argv[1] + "/" + folder
  	sp.call(["../build/gmm_loglikelihood",path,sys.argv[2]])
