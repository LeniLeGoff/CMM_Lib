#!/usr/bin/env python

import matplotlib.pyplot as plt
import sys
import os
import numpy as np

def load_class_stat(file_path):
  epoch = list()
  nbr_comp = list()
  nbr_samples = list()
  # errors = list()
  for i in range(0,10) :
    nbr_comp.append(list())
    nbr_samples.append(list())
    # errors.append(list())

  with open(file_path) as file :
    content = file.readlines()
    epoch_num = 0
    for line in content :
      linesplit = line.split(" ")
      if linesplit[0] == "EPOCH" :
        epoch_num = int(linesplit[2])
        if epoch_num%10 == 0 :
          epoch.append(epoch_num)
      linesplit = line.split(" : ")
      if linesplit[0].split(" ")[0] == "class" and epoch_num%10 == 0 :
        index = int(linesplit[0].split(" ")[1])
        nbr_comp[index].append(int(linesplit[1]))
        nbr_samples[index].append(int(linesplit[2]))

        # errors[index].append(float(linesplit[3]))
  
  return epoch, nbr_comp, nbr_samples

def load_error(file_path):
  error = list()
  epoch = list()
  with open(file_path) as file :
    content = file.readlines()
    for line in content :
      linesplit = line.split(" ")
      if linesplit[0] == "ERROR" :
        error.append(float(linesplit[2]))
      elif linesplit[0] == "EPOCH" :
        epoch.append(int(linesplit[2]))

  return epoch, error

