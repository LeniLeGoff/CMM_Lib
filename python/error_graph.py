#!/usr/bin/env python

import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import data_loaders as dl


plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['figure.edgecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.edgecolor'] = 'white'

if len(sys.argv) != 4 :
  print("Usage : \narg1 : folder path")
  print("arg2 : online or batch")
  print("arg3 : param name")
  sys.exit(1)

errors_v = list()
x = list()
param_value = list()
for folder in os.listdir(sys.argv[1]) :
  folder_path = sys.argv[1] + "/" + folder
  errors = list()
  for file in os.listdir(folder_path) :
    file_path = folder_path + "/" + file
    if(sys.argv[2] == "batch"):
      param_value.append(float(file.split("_")[-1].split("log")[0][:-1]))#split(".")[0]))
      x, error = dl.load_batch_error(file_path)
    elif (sys.argv[2] == "online"):
      x, error = dl.load_online_error(file_path)
      param_value.append(float(file.split("_")[-1].split("-")[0]))


    errors.append(error)
  errors_v.append(errors)

param_value = param_value[:len(errors_v[0])]


errors_reorg = list()
for j in range(0,len(errors_v[0])) :
  errors2 = list()
  for i in range(0,len(errors_v)) :
    errors2.append(errors_v[i][j])
  errors_reorg.append(errors2)  

errors_m = list()
errors_75p = list()
errors_25p = list()
for error in errors_reorg :
  e = np.zeros((len(error),len(x)))
  for i in range(0,len(error)) :
    e[i, :] = error[i]
  m, third_perc, first_perc = dl.median_perc(e)
  errors_m.append(m)
  errors_75p.append(third_perc)
  errors_25p.append(first_perc)



indexes = sorted(range(len(param_value)), key=lambda k: param_value[k])
param_value.sort()
errors_sorted = list()
errors_sorted_75p = list()
errors_sorted_25p = list()
for i in indexes :
  errors_sorted.append(errors_m[i])
  errors_sorted_75p.append(errors_75p[i])
  errors_sorted_25p.append(errors_25p[i])
errors_m = errors_sorted
errors_75p = errors_sorted_75p
errors_25p = errors_sorted_25p

cmap = plt.get_cmap("gnuplot")
colors =  [cmap(i) for i in np.linspace(0,1,len(errors_m))]

fig, ax1 = plt.subplots(1,sharex=True)
ax1.set_ylim([0.,0.6])


for i in range(0,len(errors_m)) :
  ax1.plot(x,errors_m[i],linewidth=2,color=colors[i],label= sys.argv[3] + " = " + str(param_value[i]))
  ax1.fill_between(x,errors_25p[i],errors_m[i],facecolor=colors[i],alpha=0.2)
  ax1.fill_between(x,errors_75p[i],errors_m[i],facecolor=colors[i],alpha=0.2)
  ax1.text(x[-1],errors_m[i][-1],str(param_value[i]))



plt.legend(bbox_to_anchor=(0., 1., 1., .0), loc=0,ncol=6)#, borderaxespad=0.)

folder = sys.argv[1]

print(folder)

exp_name = folder.split("/")[-2]

print(exp_name)

plt.tight_layout()

# plt.savefig(sys.argv[1] + "../graphs/"  + exp_name + ".png")#,bbox_inches='tight')


plt.show()
