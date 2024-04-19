import numpy as np
import spams
import pickle
import os
import time

import settings

Clustering = True
Randomization = False
lambda_v = 0.001
method = 'full'
dataset_dir = settings.results_dir + settings.datasets[settings.dataset_index] + '_lamba' + str(lambda_v) + '/'
method_dir = dataset_dir + method + '/'
cnn_res_dir = method_dir + settings.CNNs[settings.net_indx] + '/'
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)
if not os.path.exists(method_dir):
    os.mkdir(method_dir)
if not os.path.exists(cnn_res_dir):
    os.mkdir(cnn_res_dir)

activations = None
labels = None
for class_index in range(1000):
    with open('./project_antwerp/dataset/Activationmaps/vebi/' + settings.CNNs[settings.net_indx] + '/' +
              str(class_index) + '.npy', 'rb') as file_add:
        class_info = pickle.load(file_add)
    if activations is None:
        activations = class_info[0]  # average of activations
        labels = class_info[1]  # binary representation of label
    else:
        activations = np.concatenate((activations, class_info[0]), axis=0)
        labels = np.concatenate((labels, class_info[1]), axis=0)

indices = np.arange(activations.shape[0])
np.random.shuffle(indices)

activations = activations[indices, :]
labels = labels[indices, :]

X = np.asfortranarray(activations)
L = np.asfortranarray(labels)

w = spams.lasso(X=L, D=X, lambda1=lambda_v, mode=0, pos=True, numThreads=5)
w = np.swapaxes(w.toarray(), 0, 1)

with open(cnn_res_dir + 'w' + '.npy', 'wb') as file_add:
        pickle.dump(w, file_add)


