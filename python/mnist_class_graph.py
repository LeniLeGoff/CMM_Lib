#!/usr/bin/env python

import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import mnist_graph as mnist



if len(sys.argv) != 2 :
  print("Usage : \narg1 : folder path")


epoch = list()
param_value = list()
nbr_comp_v = list()
nbr_samples_v = list()
errors_v = list()

for file in os.listdir(sys.argv[1]) :
  param_value.append(float(file.split("_")[-1].split("log")[0][:-1]))
  file_path = sys.argv[1] + "/" + file
  epoch, nbr_comp, nbr_samples = mnist.load_class_stat(file_path)
  nbr_comp_v.append(nbr_comp)
  nbr_samples_v.append(nbr_samples)

indexes = sorted(range(len(param_value)), key=lambda k: param_value[k])
param_value.sort()
ncomp_sorted = list()
nsamples_sorted = list()
errors_sorted = list()
for i in indexes :
  ncomp_sorted.append(nbr_comp_v[i])
  nsamples_sorted.append(nbr_samples_v[i])
  # errors_sorted.append(errors[i])
nbr_comp_v = ncomp_sorted
nbr_samples_v = nsamples_sorted
# errors = errors_sorted

cmap = plt.get_cmap("gnuplot")
colors = [cmap(i) for i in np.linspace(0,1,len(nbr_comp_v))]

fig, ax1 = plt.subplots(1,sharex=True)
for i in range(0,len(nbr_comp_v)) :
  for k in range(0,len(nbr_comp_v[i])):
      x = np.array(epoch) + k
      ax1.bar(x,nbr_comp_v[i][k],1.,color=colors[i],label="alpha = " + str(param_value[i]))


plt.show()