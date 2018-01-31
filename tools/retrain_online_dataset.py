#!\usr\bin\env python

import os
import sys
import subprocess as sp

for dataset_file in os.listdir(sys.argv[1]) :
	print(dataset_file)
	newfolder = "retrain_" + dataset_file.split(".")[0]
	sp.call(["../build/online_train_gmm",sys.argv[1]+"/"+dataset_file,newfolder])

