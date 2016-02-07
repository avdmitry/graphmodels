mat_path = "./ILSVRC2014_devkit/data/meta_clsloc.mat"

labels_file = open("labels.txt", "w")

import scipy.io
meta_mat = scipy.io.loadmat(mat_path)
synsets = meta_mat['synsets'][0]
for synset in synsets:
    labels_file.write("{0},{1},{2}\n".format(synset[0][0][0]-1, synset[1][0], synset[2][0]))

labels_file.close()
