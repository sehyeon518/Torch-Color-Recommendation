from glob import glob
import json
import os
import numpy as np


def get_top_4_colors(probability_arr):
    top_indices = np.argsort(probability_arr[0])[-4:][::-1]

    return top_indices


def get_top_indices_and_probabilities(output, list_of_colors, num_indices=4):
    top_indices = np.argsort(output)[::-1][:num_indices]

    top_probabilities = output[top_indices]

    normalized_probabilities = top_probabilities / np.sum(top_probabilities)
    
    sorted_indices = np.argsort(normalized_probabilities)[::-1]
    sorted_probabilities = normalized_probabilities[sorted_indices]

    top_colors = [list_of_colors[idx] for idx in top_indices[sorted_indices]]

    return top_colors, sorted_probabilities


def load_templates_features():
    id_arr, colors_arr, weights_arr = [], [], []
    
    total_json = []

    fns = glob("./features/*.json")
    for fn in fns:
        with open(fn, 'r') as rf:
            total_json.extend(json.load(rf))
    
    for data in total_json:
        id_arr.append(data['id'])
        colors_arr.append(np.array(data['colors']).flatten())
        weights_arr.append(np.array(data['weights']).flatten())
    
    return np.array(id_arr), np.array(colors_arr), np.array(weights_arr)


def calculate_cos_similarity(target, data_arr):
    target = np.array(target) / np.linalg.norm(target)
    data_arr = np.array(data_arr) / np.linalg.norm(data_arr, axis=1)[:,None]
    cos_sim = np.matmul(target, data_arr.T)[0]

    cos_sim[np.isnan(cos_sim)] = 0
    return cos_sim


def get_close_index(similarity, id_arr, max_num=None):
    top_n_index = similarity.argsort()[::-1]
    if max_num is not None and max_num > 0:
        top_n_index = top_n_index[:max_num]
    return [int(id) for id in np.array(id_arr)[top_n_index]]


def extract_euclidien_similarity(data_arr):
    data_arr = np.array(data_arr)
    norm_data = np.sum(data_arr ** 2, axis=1).reshape(-1, 1)
    squared_distances = norm_data + norm_data.T - 2 * np.dot(data_arr, data_arr.T)
    squared_distances = np.maximum(squared_distances, 0)
    distances = np.sqrt(squared_distances)
    similarities = 1 / (1 + distances)
    np.fill_diagonal(similarities, 1)
    
    return similarities


def get_template_dict():
    template_paths = glob("/media/mlfavorfit/sdb/template/*/*.jpg")
    template_dict = {int(os.path.basename(path).split(".")[0]):path for path in template_paths}
    return template_dict

def concat_array(arr1, arr2, axis=0):
    return np.concatenate([arr1, arr2], axis=axis)
