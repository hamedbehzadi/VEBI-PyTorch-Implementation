import numpy as np
import spams
import pickle
import os
import time

from InterpretationViaModelInversion.VEBI import settings

Clustering = True
Randomization = False

lambda_v = 0.001
# d2pruning_classcenter
# moderate_selection_threequantile
cluster_method = 'moderate_selection'
if Clustering:
    method = 'ours'
    dataset_dir = settings.results_dir + settings.datasets[settings.dataset_index] + '_lambda' + str(lambda_v) + '/' + cluster_method + '/'
    method_dir = dataset_dir + method + '/'
    cnn_res_dir = method_dir + settings.CNNs[settings.net_indx] + '/'
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    if not os.path.exists(method_dir):
        os.mkdir(method_dir)
    if not os.path.exists(cnn_res_dir):
        os.mkdir(cnn_res_dir)
    with open('./project_antwerp/dataset/Activationmaps/' + settings.CNNs[settings.net_indx] + '_' + cluster_method
              + '_info.npy',
              'rb') as file_add:
        cluster_info = pickle.load(file_add)
    data_percentage = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    data_percentage = [0.95]
    for percentage in data_percentage:
        activations = None
        labels = None
        for class_index in range(1000):
            data_indices = cluster_info[class_index][percentage]['coreset_incdices']

            with open('./project_antwerp/dataset/Activationmaps/vebi/' + settings.CNNs[settings.net_indx] + '/' +
                      str(class_index) + '.npy', 'rb') as file_add:
                class_info = pickle.load(file_add)
            if activations is None:
                activations = class_info[0][data_indices, :]
                labels = np.asarray(class_info[1])[data_indices, :]
            else:
                activations = np.concatenate((activations, class_info[0][data_indices, :]), axis=0)
                labels = np.concatenate((labels, np.asarray(class_info[1])[data_indices, :]), axis=0)

        indices = np.arange(activations.shape[0])
        np.random.shuffle(indices)

        activations = activations[indices, :]
        labels = labels[indices, :]
        
        X = np.asfortranarray(activations)
        L = np.asfortranarray(labels)

        del activations
        del labels

        w = spams.lasso(X=L, D=X, lambda1=lambda_v, mode=0, pos=True, numThreads=5)
        w = np.swapaxes(w.toarray(), 0, 1)

        with open(cnn_res_dir + 'w_' + str(percentage) + '.npy', 'wb') as file_add:
            pickle.dump(w, file_add)

        del X
        del L
        del w

elif Randomization:
    method = 'Random'
    dataset_dir = settings.results_dir + settings.datasets[settings.dataset_index] + '_lambda' + str(lambda_v) + '/'
    method_dir = dataset_dir + method + '/'
    cnn_res_dir = method_dir + settings.CNNs[settings.net_indx] + '/'
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    if not os.path.exists(method_dir):
        os.mkdir(method_dir)
    if not os.path.exists(cnn_res_dir):
        os.mkdir(cnn_res_dir)
    with open('./project_antwerp/dataset/Activationmaps/class_length.npy', 'rb') as file_add:
        class_length = pickle.load(file_add)
    with open('./project_antwerp/dataset/Activationmaps/' + settings.CNNs[settings.net_indx] + '_'+ cluster_method + '_info.npy',
              'rb') as file_add:
        cluster_info = pickle.load(file_add)
    data_percentage = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    data_percentage = [0.95]
    for percentage in data_percentage:
        activations = None
        labels = None
        for class_index in range(1000):
            num_centers = len(cluster_info[class_index][percentage]['coreset_incdices'])
            data_indices = np.sort(
                np.random.choice(range(0, int(class_length[class_index])), num_centers, replace=False))
            with open('./project_antwerp/dataset/Activationmaps/vebi/' + settings.CNNs[settings.net_indx] + '/' +
                      str(class_index) + '.npy', 'rb') as file_add:
                class_info = pickle.load(file_add)
            if activations is None:
                activations = class_info[0][data_indices, :]
                labels = np.asarray(class_info[1])[data_indices, :]
            else:
                activations = np.concatenate((activations, class_info[0][data_indices, :]), axis=0)
                labels = np.concatenate((labels, np.asarray(class_info[1])[data_indices, :]), axis=0)

        indices = np.arange(activations.shape[0])
        np.random.shuffle(indices)

        activations = activations[indices, :]
        labels = labels[indices, :]

        X = np.asfortranarray(activations)
        L = np.asfortranarray(labels)

        del activations
        del labels

        w = spams.lasso(X=L, D=X, lambda1=lambda_v, mode=0, pos=True, numThreads=5)
        w = np.swapaxes(w.toarray(), 0, 1)

        with open(cnn_res_dir + 'w_' + str(percentage) + '.npy', 'wb') as file_add:
            pickle.dump(w, file_add)

        del X
        del L
        del w
else:
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


