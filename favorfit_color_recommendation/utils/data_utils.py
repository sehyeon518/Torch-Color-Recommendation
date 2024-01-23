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