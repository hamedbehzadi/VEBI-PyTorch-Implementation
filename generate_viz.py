import numpy as np
import pickle
import os

import torch
from torchvision import models, datasets
from torch.utils.data import DataLoader

from InterpretationViaModelInversion.VEBI import rep_sim_helper
from InterpretationViaModelInversion.VEBI import settings
from InterpretationViaModelInversion.VEBI import generate_heatmap as gh


class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, item):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(item)
        path = self.imgs[item][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class modifying_activationmaps:
    def __init__(self):
        self.current_activationmaps = []
        self.length = []
        self.weight = []
        self.layer_counter = 0
    def modf_activations(self, module, input, output):
        self.current_activationmaps.append(output)
        return output

def register_hook_modf_act(module):
    name = module.__class__.__name__
    if name.find('ReLU') != -1:
        module.register_forward_hook(acts_obj.modf_activations)

acts_obj = modifying_activationmaps()


lamda = '_lambda0.001'
dataset_dir = settings.results_dir + settings.datasets[settings.dataset_index] + lamda + '/'

if settings.CNNs[settings.net_indx] == 'resnet50':
    model_cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    transformation = models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    model_cnn.apply(register_hook_modf_act)
elif settings.CNNs[settings.net_indx] == 'resnet18':
    model_cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    transformation = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    model_cnn.apply(register_hook_modf_act)
elif settings.CNNs[settings.net_indx] == 'vgg19':
    model_cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    transformation = models.VGG19_Weights.IMAGENET1K_V1.transforms()
    model_cnn.features.apply(register_hook_modf_act)

with open(dataset_dir + settings.CNNs[settings.net_indx] + '_ReluLength.npy', 'rb') as file_add:
    acts_obj.length = pickle.load(file_add)

model_cnn.eval()
#model_cnn.cuda()

val_dataset_dir = ImageFolderWithPaths('./project_antwerp/dataset/ImageNet/validation_category/', transformation)
val_loader = torch.utils.data.DataLoader(val_dataset_dir, batch_size=50, shuffle=False)

first_method = 'full'
# d2pruning_classcenter --- moderate_selection_threequantile --- moderate_selection
second_method = 'd2pruning_classcenter'

first_method_dir = dataset_dir + first_method + '/'
first_cnn_res_dir = first_method_dir + settings.CNNs[settings.net_indx] + '/'
with open(first_cnn_res_dir + 'w.npy', 'rb') as file_add:
    first_w = pickle.load(file_add)

if second_method == 'Random':
    seconf_method_dir = dataset_dir + second_method + '/'
    second_cnn_res_dir = seconf_method_dir + settings.CNNs[settings.net_indx] + '/'
else:
    seconf_method_dir = dataset_dir + second_method + '/ours/'
    second_cnn_res_dir = seconf_method_dir + settings.CNNs[settings.net_indx] + '/'

data_percentage = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.95]
for percentage in data_percentage:
    Distance = []

    if second_method == 'Random':
        vis_address = ('./project_antwerp/InterpretationViaModelInversion/VEBI/results/' +
                       settings.datasets[settings.dataset_index] + '_lambda0.001/Random/visualization/')
        if not os.path.exists(vis_address):
            os.mkdir(vis_address)
        vis_address = vis_address + settings.CNNs[settings.net_indx] + '/'
        if not os.path.exists(vis_address):
            os.mkdir(vis_address)
        vis_address = vis_address + str(percentage) + '/'
        if not os.path.exists(vis_address):
            os.mkdir(vis_address)
    else:
        vis_address = ('./project_antwerp/InterpretationViaModelInversion/VEBI/results/' +
                       settings.datasets[settings.dataset_index] + '_lambda0.001' +
                       '/' + second_method + '/ours/visualization/')
        if not os.path.exists(vis_address):
            os.mkdir(vis_address)
        vis_address = vis_address + settings.CNNs[settings.net_indx] + '/'
        if not os.path.exists(vis_address):
            os.mkdir(vis_address)
        vis_address = vis_address + str(percentage) + '/'
        if not os.path.exists(vis_address):
            os.mkdir(vis_address)

    selectet_class_indexes = np.arange(0, 1000)
    with open(second_cnn_res_dir + 'w_' + str(percentage) + '.npy', 'rb') as file_add:
        second_w = pickle.load(file_add)

    for class_index, (inputs, labels, paths) in enumerate(val_loader):
        if class_index in selectet_class_indexes:
            outputs = model_cnn(inputs)

            pred_label = torch.argmax(outputs, dim=1)
            correctly_classified_indices = torch.where(pred_label == labels)[0]
            temp_featuremaps = acts_obj.current_activationmaps
            featuremaps = []
            for layer_featuremaps in temp_featuremaps:
                featuremaps.append(layer_featuremaps[:,:,:,:])
            #correctly_classified_indices = correctly_classified_indices.detach().cpu().numpy()
            #print(correctly_classified_indices)
            #paths = np.array(paths)[correctly_classified_indices]
            first_w_class = first_w[class_index, :]
            second_w_class = second_w[class_index, :]

            layers_analysis = rep_sim_helper.check_indices_4_vis(acts_obj.length,first_w_class,second_w_class)
            for layer_info in layers_analysis:
                if layer_info[0] == 1:
                    print(class_index,layer_info)
            #print('finished')
            #exit()
            #layers_analysis.reverse()
            #last_layer = []
            #last_layer.append(layers_analysis[1])
            #gh.generate_heatmap(featuremaps,paths,layers_analysis,str(class_index),vis_address)

            acts_obj.current_activationmaps = []
            del temp_featuremaps
            del featuremaps

