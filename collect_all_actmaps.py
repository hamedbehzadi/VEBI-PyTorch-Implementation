import torchvision.models as models
import torch
import pickle
import numpy as np

from InterpretationViaModelInversion.lmdbDataset import LMDBDataset


class collecting_activationmaps:
    def __init__(self):
        self.current_activationmaps = []
    def collect_activations(self, module, input, output):
        self.current_activationmaps.append(output)
        del output
        del input
        del module


acts_obj = collecting_activationmaps()

def register_hook(module):
    name = module.__class__.__name__
    if name.find('ReLU') != -1:
        module.register_forward_hook(acts_obj.collect_activations)



def generating_l2_norm(all_fmaps,samples_num):
    l2_norm = None
    for sample_indx in range(samples_num):
        l2_norm_temp_sample = None
        for i in range(len(all_fmaps)):
            featuremaps = torch.unsqueeze(all_fmaps[i][sample_indx],dim=0)
            fmap_squared = torch.pow(featuremaps, 2)
            fmap_squared_sum_perfiltr = torch.sum(fmap_squared, (2, 3))
            fmap_sum_allfilter = torch.sum(fmap_squared)
            fmap_normalize = torch.div(fmap_squared_sum_perfiltr, fmap_sum_allfilter).detach().cpu().numpy()

            if l2_norm_temp_sample is None:
                l2_norm_temp_sample = fmap_normalize
            else:
                l2_norm_temp_sample = np.concatenate((l2_norm_temp_sample, fmap_normalize), axis=1)
        if l2_norm is None:
            l2_norm = l2_norm_temp_sample
        else:
            l2_norm = np.concatenate((l2_norm, l2_norm_temp_sample), axis=0)
    return l2_norm

# Resnet50
model_cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
transformation = models.ResNet50_Weights.IMAGENET1K_V1.transforms()
model_cnn.apply(register_hook)

# Resnet18
#model_cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#transformation = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
#model_cnn.apply(register_hook)

# VGG19
#model_cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
#transformation = models.VGG19_Weights.IMAGENET1K_V1.transforms()
#model_cnn.features.apply(register_hook)

model_cnn.cuda()
model_cnn.eval()

emb = torch.nn.Embedding(1000, 1000)
emb.weight.data = torch.eye(1000)

with open('project_antwerp/dataset/Activationmaps/class_length.npy','rb') as file_add:
    class_length = pickle.load(file_add)
    
dataset = LMDBDataset('./datasets_antwerp/ImageNet/train.lmdb', transform=transformation)

for class_index in range(1000):
    counter = 0
    if class_index > 0:
        for i in range(class_index):
            counter += class_length[i]

    indices = np.arange(int(counter), int(counter + class_length[class_index]))
    new_dataset = torch.utils.data.Subset(dataset, indices)
    train_loader = torch.utils.data.DataLoader(new_dataset, batch_size=350, shuffle=False,
                                               num_workers=12)

    real_labels = None
    pred_labels = None
    real_logits = []
    pred_logits = []
    X = None
    L = []
    for inputs, labels in train_loader:
        outputlogit = model_cnn(inputs.cuda()).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        pred = np.argmax(outputlogit, axis=1)
        l2_norm = generating_l2_norm(acts_obj.current_activationmaps,labels.shape[0])

        if X is None:
            X = l2_norm
        else:
            X = np.concatenate((X,l2_norm),axis=0)
        if real_labels is None:
            real_labels = labels
            pred_labels = pred
        else:
            real_labels = np.concatenate((real_labels, labels), axis=0)
            pred_labels = np.concatenate((pred_labels, pred), axis=0)
        for i in range(labels.shape[0]):
            binary_label = emb(torch.from_numpy(np.asarray(pred[i]))).detach().cpu().numpy()
            L.append(binary_label)
            real_logits.append(outputlogit[i, labels[i]])
            pred_logits.append(outputlogit[i, pred[i]])

        acts_obj.current_activationmaps = []


    with open('./project_antwerp/dataset/Activationmaps/vebi/resnet50_1/' + str(class_index) + '.npy', 'wb') as file_add:
        pickle.dump((X, L, real_labels, pred_labels, real_logits, pred_logits), file_add)

    del X
    del L
    del real_labels
    del pred_labels
    del real_logits
    del pred_logits