#!\usr\bin\env python

import sys
import os
import subprocess as sp

if len(sys.argv) != 2:
	print("Usage : folder of archive experiments")
  	sys.exit(1)

for folder in os.listdir(sys.argv[1]) :
	path = sys.argv[1] + "/" + folder
	dataset_path = path + "/iteration_399/dataset_meanFPFHLabHist.yml" 
  	sp.call(["../build/gmm_loglikelihood",path,dataset_path])
