#!/usr/bin/env python

import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import mnist_graph as mnist



if len(sys.argv) != 2 :
  print("Usage : \narg1 : folder path")

errors = list()
epoch = list()
param_value = list()
for file in os.listdir(sys.argv[1]) :
  param_value.append(float(file.split("_")[-1].split("log")[0][:-1]))
  file_path = sys.argv[1] + "/" + file
  epoch, error = mnist.load_error(file_path)
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
  ax1.plot(epoch,errors[i],linewidth=2,color=colors[i],label="alpha = " + str(param_value[i]))
  ax1.text(300,errors[i][-1],str(param_value[i]))

plt.legend()

plt.show()