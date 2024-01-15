import json
import numpy as np
import itertools
from colorspace_converter import rgb_to_lab
from calculate_statistics import *
import colorsys

# TODO file path
load_json_file = r'C:\Users\mlfav\lib\shlee\colors.jsonl'
json_file_path = r'C:\Users\mlfav\lib\shlee\color_model\ColorModel\train.jsonl'
count = 0

with open(load_json_file, 'r') as f:
    for line in f:
        line = json.loads(line)

        color_palette = line['colors']

        RGB_8 = np.empty((8, 3), dtype=float)
        HSV_8 = np.empty((8, 3), dtype=float)
        LAB_8 = np.empty((8, 3), dtype=float)
        for idx, color in enumerate(color_palette):
            r, g, b = color
            r, g, b = r / 255.0, g / 255.0, b / 255.0
            RGB_8[idx] = np.array([r, g, b]).reshape((1, 3))
            HSV_8[idx] = np.array(list(colorsys.rgb_to_hsv(r, g, b))).reshape((1, 3))
            LAB_8[idx] = np.array([rgb_to_lab(r, g, b)]).reshape((1, 3))

        RGB_8 = RGB_8[np.argsort(LAB_8[:, 0])]
        HSV_8 = HSV_8[np.argsort(LAB_8[:, 0])]
        LAB_8 = LAB_8[np.argsort(LAB_8[:, 0])]

        data = {}
        RGB = RGB_8[2:6]
        HSV = HSV_8[2:6]
        LAB = LAB_8[2:6]   

        RGB_R, RGB_G, RGB_B = RGB[:, 0], RGB[:, 1], RGB[:, 2]
        HSV_H, HSV_S, HSV_V = HSV[:, 0], HSV[:, 1], HSV[:, 2]
        LAB_L, LAB_A, LAB_B = LAB[:, 0], LAB[:, 1], LAB[:, 2]

        hue_probabilities = []
        for color in HSV_H:
            probabilities = hue_probability(color, HSV_H)
            hue_probabilities.append(probabilities)

        adjacent_probabilities = []
        for color in HSV_H:
            probabilities = hue_adjacent_probability(color, HSV_H)
            adjacent_probabilities.append(probabilities)

        all_CH_values = []
        LCH_array = np.column_stack([LAB_L, LAB_A, HSV_H])
        for color1, color2 in itertools.combinations(LCH_array, 2):

            L1, a1, b1 = color1
            L2, a2, b2 = color2
            result = calculate_CH(L1, np.sqrt(a1**2 + b1**2), np.degrees(np.arctan2(b1, a1)),
                           L2, np.sqrt(a2**2 + b2**2), np.degrees(np.arctan2(b2, a2)))

            all_CH_values.append(result)

        RGB = RGB.tolist()
        HSV = HSV.tolist()
        CIELAB = LAB.tolist()
        RGB_feature = [calculate_statistics(RGB_R), calculate_statistics(RGB_G), calculate_statistics(RGB_B)]
        HSV_feature = [calculate_statistics(HSV_H), calculate_statistics(HSV_S), calculate_statistics(HSV_V)]
        LAB_feature = [calculate_statistics(LAB_L), calculate_statistics(LAB_A), calculate_statistics(LAB_B)]
        Hue_probability = [calculate_hsv_statistics(hue_probabilities)]
        Hue_Log_probability = [calculate_hsv_statistics(hue_probabilities, True)]
        Hue_adjacent_probability = [calculate_hsv_statistics(adjacent_probabilities)]
        Hue_Log_adjacent_probability = [calculate_hsv_statistics(adjacent_probabilities, True)]
        Hue_entropy = [[calculate_hue_entropy(HSV_H)]]
        CH = [all_CH_values]
        Light_gradient = [calculate_gradient(LAB_L).tolist()]
        Hue_gradient = [calculate_gradient(HSV_H).tolist()]
        data['input_data'] = sum((RGB, HSV, CIELAB, RGB_feature, HSV_feature, LAB_feature, Hue_probability, Hue_Log_probability, Hue_adjacent_probability, Hue_Log_adjacent_probability, Hue_entropy, CH, Light_gradient, Hue_gradient), [])
        data['input_data'] = [[element for sublist in data['input_data'] for element in sublist]]

        data['output_color'] = HSV_8[0:2].tolist() + HSV_8[6:8].tolist()

        with open(json_file_path, 'a') as jsonl_file:
            json.dump(data, jsonl_file)
            jsonl_file.write('\n')

        if count % 100 == 0:
            print("count:", count)
        count += 1
        
print("done:", count)
