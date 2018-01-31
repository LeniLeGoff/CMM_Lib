#!/usr/bin/env python

import yaml
import sys
import os

def load_dataset(path) :

    data = list()
    labels = list()

    with open(path) as FILE:
        yaml_file = yaml.load(FILE)
        for sample in yaml_file["frame_0"]["features"] :
            is_nan = False
            vect = yaml_file["frame_0"]["features"][sample]["value"]
            for val in vect :
                if(val == 'nan' or val == '-nan'):
                    is_nan = True
            if(is_nan):
                continue
            labels.append(int(yaml_file["frame_0"]["features"][sample]["label"]))
            data.append(vect)

    return data, labels

if len(sys.argv) != 2 :
  print("Usage : folder with data yaml files")
  sys.exit(1)

data_list = list()
label_list = list()
for data_file in os.listdir(sys.argv[1]) :
    data, label = load_dataset(sys.argv[1] + "/" + data_file)
    data_list.append(data)
    label_list.append(label)

pos_data = list()
neg_data = list()
for i in range(0,len(data_list)) :
  for j in range(0,len(data_list[i])) :
    if label_list[i][j] == 1 :
      pos_data.append(data_list[i][j])
    else :
      neg_data.append(data_list[i][j])

print("size of pos dataset : " + str(len(pos_data)))
print("size of neg dataset : " + str(len(neg_data)))

neg_data = neg_data[:len(pos_data)]

output_file = sys.argv[1] + "/merged_dataset.yml"

dico = dict()
j = 0
for i in range(0,len(pos_data)) :
  dico["feature_" + str(j)] = {"label" : 1, "value" : pos_data[i]}
  j+=1
  dico["feature_" + str(j)] = {"label" : 0, "value" : neg_data[i]}
  j+=1

yaml_dic = {"frame_0" : {"timestamp" : {"sec" : 0, "nsec": 0}, "features" : dico}}
with open(output_file,"w") as file :
  yaml.dump(yaml_dic,file)






