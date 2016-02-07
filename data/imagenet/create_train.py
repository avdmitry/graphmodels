train_path = "./ilsvrc/train/"

import os

labels = {}
labels_file = open("labels.txt", "r")
for line in labels_file:
  label = line.split(",")[0]
  synset = line.split(",")[1]
  labels[synset] = label
labels_file.close()

train_file = open("train.txt", "w")
for synset in os.listdir(train_path):
  synset_path = train_path + synset + "/"
  label = labels[synset]
  for image in os.listdir(synset_path):
    train_file.write(synset_path + image + " " + label + "\n")
train_file.close()
