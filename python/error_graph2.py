#!/usr/bin/env python

import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import data_loaders as dl



if len(sys.argv) != 3 :
  print("Usage : \narg1 : folder path")
  print("arg2 : online or batch")
  sys.exit(1)

errors = list()
x = list()
param_value = list()
for file in os.listdir(sys.argv[1]) :
  param_value.append(float(file.split("_")[-1].split("-")[0]))
  file_path = sys.argv[1] + "/" + file
  if(sys.argv[2] == "batch"):
  	x, error = dl.load_batch_error(file_path)
  elif (sys.argv[2] == "online"):
  	x, error = dl.load_online_error(file_path)

  errors.append(error)

indexes = sorted(range(len(param_value)), key=lambda k: param_value[k])
param_value.sort()
errors_sorted = list()
for i in indexes :
  errors_sorted.append(errors[i])
errors = errors_sorted

cmap = plt.get_cmap("gnuplot")
colors = [cmap(i) for i in np.linspace(0,1,len(errors))]

fig, ax1 = plt.subplots(1,sharex=True)
for i in range(0,len(errors)) :
  ax1.plot(x,errors[i],linewidth=2,color=colors[i],label="alpha = " + str(param_value[i]))
  ax1.text(x[-1],errors[i][-1],str(param_value[i]))

plt.legend()

plt.show()