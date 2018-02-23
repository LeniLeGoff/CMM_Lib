#!\usr\bin\env python

import sys
import os
import subprocess as sp


if len(sys.argv) != 2:
	print("Usage : archive folder of an experiment")
  	sys.exit(1)

gmm_file = ""
dataset_file = ""

for iter_folder in os.listdir(sys.argv[1]) :
	if(iter_folder.split("_")[0] == "iteration") :
		for file in os.listdir(sys.argv[1] + "/" + iter_folder) :
			if(file.split("_")[0] == "gmm"):
				gmm_file = file
			if(file.split("_")[0] == "dataset") :
				dataset_file = file
		gmm_file = sys.argv[1] + "/" + iter_folder + "/" + gmm_file
		dataset_file = sys.argv[1] + "/" + iter_folder + "/" + dataset_file
		print(gmm_file)
		sp.call(["../build/gen_archive_stat",gmm_file,dataset_file])
