import os.path

import cv2
import torch
import numpy as np


def write_image(write_add, A):
    cv2.imwrite(write_add, np.uint8(A * 255))


def superimposing(image, heatmap):
    vis = cv2.addWeighted(heatmap, 0.6, image, 0.4, 0)
    return vis / np.max(vis)


def apply_colormap(A):
    A = cv2.applyColorMap(np.uint8(255 * A), cv2.COLORMAP_JET)
    return np.float32(A) / 255


def normalize_numpy(A):
    if np.max(A) == 0.0:
        return A
    return (A - np.min(A)) / (np.max(A) - np.min(A))


def read_image(image_address, shape):
    image_name = image_address.split('/')[-1]
    image = cv2.imread(image_address)
    image = cv2.resize(image, dsize=(shape, shape))
    image = np.float32(image)
    image = normalize_numpy(image)
    image = image[:, :, ::-1]
    return image, image_name


def generate_heatmap(featuremaps, paths, layer_analyis, class_index, vis_address):
    for layer_index, layer_info in enumerate(layer_analyis):
        if layer_info[0] == 1:
            channel_indices = layer_info[1]
            for i in range(len(paths)):
                image, image_name = read_image(paths[i], 224)
                for channel_index in channel_indices:
                    feature_map = featuremaps[layer_index][i, channel_index , :, :].detach().cpu().numpy()
                    feature_map = cv2.resize(feature_map, dsize=(224, 224))
                    feature_map = normalize_numpy(feature_map)
                    heatmap = apply_colormap(feature_map)
                    vis = superimposing(image, heatmap)
                    if not os.path.exists(vis_address + class_index):
                        os.mkdir(vis_address + class_index)
                    saved_add = (vis_address + class_index + '/' + image_name +
                                 '_' + str(layer_index) + '_' + str(channel_index) + '.jpg')
                    write_image(saved_add, vis)



